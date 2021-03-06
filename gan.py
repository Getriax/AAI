'''
Plik zawiera model GAN do generowania muzyki.
Proces rozpoczyna się od wczytania plików midi, które parsowane są na wartosci liczbowych.
Model generatora przyjmuje pewną liczbę nut (seq_len=24) a jego wynikiem jest kolejna nuta.
Model dyskryminatora przyjmuje sekwencję nut i zwraca prawdopodobieństwo czy wprowadzona sekwencja zawiera nutę wygenerowaną przez generator czy jest prawdziwa

Model generatora wykorzystuje dwie komórki lstm, oraz funkcje liniowe do generowania następnej nuty
Model dyskryminatora wykorzystuje lstm
'''

from collections import OrderedDict

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from common import generate_sample_song
from constants import BATCH_SIZE

from songs_data import SongsDataModule


class NextNoteGenerator(nn.Module):
    def __init__(self, seq_len=24, num_feats=3, num_pitches=128, hidden_units=256, drop_prob=0.4, ):
        super().__init__()
        self.hidden_dim = hidden_units

        self.flat = nn.Flatten(1)
        self.fc_layer1 = nn.Linear(in_features=(seq_len * num_feats), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=2)
        self.fc_pitches = nn.Linear(in_features=hidden_units, out_features=num_pitches)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, states):
        '''
        :param x - notes batches (batch_size, seq_num, num_feats)
        :param states: - lstm hidden states
        :return: next note for each of the batches (batch_size, num_feats)
        '''

        state1, state2 = states

        x = self.flat(x)  # x of shape (batch_size, seq_num, num_feats) -> (batch_size, seq_num * num_feats)
        x = self.fc_layer1(x)
        x = F.relu(x)
        x1, c1 = self.lstm_cell1(x, state1)
        x = self.dropout(x1)
        x2, c2 = self.lstm_cell2(x, state2)
        step_duration = self.fc_layer2(x2)
        pitch = self.softmax(self.fc_pitches(x2))

        state1 = (x1, c1)
        state2 = (x2, c2)

        step = step_duration[:, 0]
        duration = step_duration[:, 1]

        return pitch, step, duration, (state1, state2)

    def init_hidden(self, batch_size=BATCH_SIZE):
        ''' Inicjalizuje stan ukryty lstm, zwracając tensor tego samego typu co waga '''
        weight = next(self.parameters()).data

        return ((weight.new(batch_size, self.hidden_dim).zero_(),
                 weight.new(batch_size, self.hidden_dim).zero_()),
                (weight.new(batch_size, self.hidden_dim).zero_(),
                 weight.new(batch_size, self.hidden_dim).zero_()))


class NoteSequenceDiscriminator(nn.Module):
    def __init__(self, num_feats=3, hidden_units=256, drop_prob=0.2, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm = nn.LSTM(input_size=num_feats, hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.fc_layer = nn.Linear(in_features=(2 * hidden_units), out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, notes, state):
        x = self.dropout(notes)
        lstm_out, state = self.lstm(x, state)
        x = self.fc_layer(lstm_out)
        x = self.sigmoid(x)

        num_dims = len(x.shape)
        reduction_dims = tuple(range(1, num_dims))

        # (N, seq_len, 0-1) -> (N, srednia 0-1)
        y = torch.mean(x, dim=reduction_dims)

        return y, state

    def init_hidden(self, batch_size=BATCH_SIZE):
        ''' Inicjalizauje stan ukryty '''
        weight = next(self.parameters()).data

        layer_mult = 2  # bo jest dwukierunkowy

        return (weight.new(self.num_layers * layer_mult, batch_size,
                           self.hidden_dim).zero_(),
                weight.new(self.num_layers * layer_mult, batch_size,
                           self.hidden_dim).zero_())


class SongsGAN(pl.LightningModule):
    def __init__(
            self,
            seq_len=24,
            num_feats=3,
            pitches_num=128,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = BATCH_SIZE,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.generator = NextNoteGenerator(seq_len=seq_len)
        self.discriminator = NoteSequenceDiscriminator(num_feats=num_feats)

        self.generator_state = self.generator.init_hidden(batch_size)
        self.discriminator_state = self.discriminator.init_hidden(batch_size)
        self.train_epoch_num = 0

    def forward(self, x, state):
        return self.generator(x, state)

    def adversarial_loss(self, y_hat, y):
        # funkcja straty GAN
        return F.binary_cross_entropy(y_hat, y)

    def prepare_notes_batch_for_generator(self, songs):
        # Ucina ostatną nutę ponieważ tutaj używany jest dataset z pełną sekwencją
        return songs[:, 0:self.seq_len]

    def append_generated_notes_to_real(self, pred_pitch, pred_step, pred_duration, actual_notes):
        '''
        Modyfikuje prawidzwe nuty (seq_len, 3) usuwając pierwszą nutę i dodając przewidziane wartosci
        '''
        actual = copy.deepcopy(actual_notes).detach()

        for i in range(0, len(actual_notes)):
            pitch = pred_pitch[i].argmax()
            step = pred_step[i]
            duration = pred_duration[i]

            # require_grad wymagany do liczenia gradientów nowego tensora, inaczej błąd
            actual[i] = torch.cat((actual[i][1:], torch.tensor([[pitch, step, duration]], requires_grad=True)))

        return actual

    def training_step(self, batch, batch_idx, optimizer_idx):
        songs = batch

        self.generator_state = self.generator.init_hidden(self.batch_size)
        self.discriminator_state = self.discriminator.init_hidden(self.batch_size)

        # trenowanie generatora
        if optimizer_idx == 0:
            x = self.prepare_notes_batch_for_generator(songs)

            # generuje następna nutę
            pitch, step, duration, gen_state = self(x, self.generator_state)
            self.generator_state = gen_state

            created = self.append_generated_notes_to_real(pitch.clone(), step.clone(), duration.clone(), x.clone())

            # generator chce oszukać dyskryminator, chcac aby zwrócił on 1 (która znaczy że jest to prawdziwa sekwencja)
            valid = torch.ones(created.size(0))
            valid = valid.type_as(songs)

            disc_valid, di_state = self.discriminator(created, self.discriminator_state)
            self.discriminator_state = di_state

            g_loss = self.adversarial_loss(disc_valid, valid)
            self.log('train_generator_loss', g_loss)
            self.log('duration_max_train', torch.max(duration), on_step=True, on_epoch=True)
            self.log('duration_min_train', torch.min(duration), on_step=True, on_epoch=True)
            self.log('step_max_train', torch.max(step), on_step=True, on_epoch=True)
            self.log('step_min_train', torch.min(step), on_step=True, on_epoch=True)

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # trenowanie dyskryminatora
        if optimizer_idx == 1:
            # Sprawdzamy jak dobrze dyksryminator radzi sobie ze sprawdzeniem czy nuta w sekwencji jest wygenerowana czy prawdziwa

            # dla prawdziwych chcemy 1
            valid = torch.ones(songs.size(0))
            valid = valid.type_as(songs)

            rel, d_state = self.discriminator(songs, self.discriminator_state)
            real_loss = self.adversarial_loss(rel, valid)

            # dla wygenerowanych 0
            fake = torch.zeros(songs.size(0))
            fake = fake.type_as(songs)

            x = self.prepare_notes_batch_for_generator(songs)
            pitch, step, duration, g_state = self(x, self.generator_state)
            self.generator_state = g_state

            g_songs = self.append_generated_notes_to_real(pitch, step, duration, x)

            d, d_state = self.discriminator(g_songs, d_state)
            self.discriminator_state = d_state

            fake_loss = self.adversarial_loss(d, fake)

            d_loss = (real_loss + fake_loss) / 2

            self.log('train_discriminator_loss', d_loss)

            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def validation_epoch_end(self, _):
        generate_sample_song(self.generator, 'gan_songs2', filename=f"song_{self.train_epoch_num}.midi", has_state=True,
                             seq_len=self.seq_len)

        self.train_epoch_num += 1

    def validation_step(self, batch, batch_idx):
        songs = batch

        self.generator_state = self.generator.init_hidden(self.batch_size)
        self.discriminator_state = self.discriminator.init_hidden(self.batch_size)

        x = self.prepare_notes_batch_for_generator(songs)

        pitch, step, duration, gen_state = self(x, self.generator_state)
        self.generator_state = gen_state

        created = self.append_generated_notes_to_real(pitch.clone(), step.clone(), duration.clone(), x.clone())

        valid = torch.ones(created.size(0))
        valid = valid.type_as(songs)

        disc_valid, di_state = self.discriminator(created, self.discriminator_state)
        self.discriminator_state = di_state

        g_loss = self.adversarial_loss(disc_valid, valid)
        self.log('val_generator_loss', g_loss)
        self.log('duration_max_val', torch.max(duration), on_step=True, on_epoch=True)
        self.log('duration_min_val', torch.min(duration), on_step=True, on_epoch=True)
        self.log('step_max_val', torch.max(step), on_step=True, on_epoch=True)
        self.log('step_min_val', torch.min(step), on_step=True, on_epoch=True)

        valid = torch.ones(songs.size(0))
        valid = valid.type_as(songs)

        rel, d_state = self.discriminator(songs, self.discriminator_state)
        real_loss = self.adversarial_loss(rel, valid)

        fake = torch.zeros(songs.size(0))
        fake = fake.type_as(songs)

        x = self.prepare_notes_batch_for_generator(songs)
        pitch, step, duration, g_state = self(x, self.generator_state)
        self.generator_state = g_state

        g_songs = self.append_generated_notes_to_real(pitch, step, duration, x)

        d, d_state = self.discriminator(g_songs, d_state)
        self.discriminator_state = d_state

        fake_loss = self.adversarial_loss(d, fake)

        d_loss = (real_loss + fake_loss) / 2

        self.log('val_discriminator_loss', d_loss)
        return d_loss


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    dm = SongsDataModule(num_train_songs=5)
    model = SongsGAN()

    logger = TensorBoardLogger("lightning_logs", name="gan model")
    trainer = Trainer(max_epochs=10, logger=logger, log_every_n_steps=1)
    trainer.fit(model, dm)

    generate_sample_song(model.generator, 'gan_songs2', 500, filename='final_song.midi', has_state=True, seq_len=24)
