from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from constants import BATCH_SIZE, INSTRUMENT_NAME

from songs_data import SongsDataModule, generate_sample_song

train_songs_len = 200
valid_songs_len = 10

instrument_note_keys = ['pitch', 'step', 'duration']

# train_songs = get_maestro_song_notes(train_songs_len)
# valid_songs = get_maestro_song_notes(valid_songs_len, skip=train_songs_len)

class Generator(nn.Module):
    def __init__(self, seq_len=24, num_feats=3, hidden_units=256, drop_prob=0.4, ):
        super().__init__()
        self.hidden_dim = hidden_units

        self.flat = nn.Flatten(1)
        self.fc_layer1 = nn.Linear(in_features=(seq_len * num_feats), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=num_feats)

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
        y = self.fc_layer2(x2)

        state1 = (x1, c1)
        state2 = (x2, c2)

        return y, (state1, state2)

    def init_hidden(self, batch_size=BATCH_SIZE):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        return ((weight.new(batch_size, self.hidden_dim).zero_(),
                 weight.new(batch_size, self.hidden_dim).zero_()),
                (weight.new(batch_size, self.hidden_dim).zero_(),
                 weight.new(batch_size, self.hidden_dim).zero_()))


class Discriminator(nn.Module):
    def __init__(self, num_feats=3, hidden_units=256, drop_prob=0.2, num_layers = 2):
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

        y = torch.mean(x, dim=reduction_dims)

        return y, state

    def init_hidden(self, batch_size=BATCH_SIZE):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 2  # for being bidirectional

        return (weight.new(self.num_layers * layer_mult, batch_size,
                           self.hidden_dim).zero_(),
                weight.new(self.num_layers * layer_mult, batch_size,
                           self.hidden_dim).zero_())


class SongsGAN(pl.LightningModule):
    def __init__(
        self,
        seq_len = 24,
        num_feats = 3,
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

        self.generator = Generator(seq_len=seq_len, num_feats=num_feats)
        self.discriminator = Discriminator(num_feats=num_feats)

        self.generator_state = self.generator.init_hidden(batch_size)
        self.discriminator_state = self.discriminator.init_hidden(batch_size)
        self.train_epoch_num = 0

    def forward(self, x, state):
        return self.generator(x, state)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prepare_notes_batch_for_generator(self, songs):
        # reduce song notes to seq_len (it comes seq_len + 1) as an actual song, next note is predicted
        return songs[:, 0:self.seq_len]

    def append_generated_notes_to_real(self, gen_notes, actual_notes, as_tensor = True):
        created = copy.deepcopy(actual_notes).detach()
        gen = gen_notes.clone()
        for i in range(0, len(gen_notes)):
            created[i] = torch.cat((created[i][1:], gen[i:(i+1)]))

            # created.append([*actual_notes[i].clone().detach().numpy(), gen_notes[i].clone().detach().numpy()])
        return created

    # def backward(
    #     self, loss, optimizer, optimizer_idx, *args, **kwargs
    # ) -> None:
    #     loss.backward(retain_graph=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        songs = batch

        self.generator_state = self.generator.init_hidden(self.batch_size)
        self.discriminator_state = self.discriminator.init_hidden(self.batch_size)

        # train generator
        if optimizer_idx == 0:

            x = self.prepare_notes_batch_for_generator(songs)

            # generate next song notes
            gen, gen_state = self.generated_notes = self(x, self.generator_state)
            self.generator_state = gen_state

            created = self.append_generated_notes_to_real(gen.clone(), x.clone())

            # ground truth result (ie: all fake)
            valid = torch.ones(created.size(0))
            valid = valid.type_as(songs)

            disc_valid, di_state = self.discriminator(created, self.discriminator_state)
            self.discriminator_state = di_state

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(disc_valid, valid)
            self.log('train_generator_loss', g_loss)

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(songs.size(0))
            valid = valid.type_as(songs)

            rel, d_state = self.discriminator(songs, self.discriminator_state)
            real_loss = self.adversarial_loss(rel, valid)

            # how well can it label as fake?
            fake = torch.zeros(songs.size(0))
            fake = fake.type_as(songs)

            x = self.prepare_notes_batch_for_generator(songs)
            g, g_state = self(x, self.generator_state)
            self.generator_state = g_state

            g_songs = self.append_generated_notes_to_real(g.detach(), x)

            d, d_state =  self.discriminator(g_songs, d_state)
            self.discriminator_state = d_state

            fake_loss = self.adversarial_loss(d, fake)

            # discriminator loss is the average of these
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

    def training_epoch_end(self, _):
        generate_sample_song(self.generator, filename=f"song_{self.train_epoch_num}.midi")

        self.train_epoch_num += 1


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    dm = SongsDataModule()
    model = SongsGAN()

    logger = TensorBoardLogger("lightning_logs", name="songs model")
    trainer = Trainer(max_epochs=5, logger=logger, log_every_n_steps=1)
    trainer.fit(model, dm)

    generate_sample_song(model.generator, 500, filename='final_song.midi')

