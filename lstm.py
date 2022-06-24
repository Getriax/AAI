import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.loggers import TensorBoardLogger

from constants import INSTRUMENT_NAME
from midi import get_random_song, notes_to_midi
from songs_data import SongsDataset, SongsDataModule


class ZeroLoss(nn.Module):
    def __init__(self, scale=False):
        self.scale = scale
        super(ZeroLoss, self).__init__()

    def forward(self, inputs):
        zeros = inputs[inputs == 0]

        if self.scale:
            return len(zeros) / len(inputs)

        return len(zeros)


class SongLTSM(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=256, num_layers=2, pitches_num=128):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fcPitch = torch.nn.Linear(hidden_size, pitches_num)
        self.fcStep = torch.nn.Linear(hidden_size, 1)
        self.fcDuration = torch.nn.Linear(hidden_size, 1)

        self.pitch_loss_function = nn.CrossEntropyLoss()
        self.step_loss_function = nn.MSELoss()
        self.step_loss_function_2 = ZeroLoss()
        self.duration_loss_function = nn.MSELoss()
        self.duration_loss_function_2 = ZeroLoss(scale=True)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.train_macro_f1 = torchmetrics.F1Score(num_classes=pitches_num, average='macro')
        self.val_macro_f1 = torchmetrics.F1Score(num_classes=pitches_num, average='macro')
        self.val_epoch_num = 0
        self.lstm_state = None

    def forward(self, x):
        x, state = self.lstm(x, self.lstm_state)
        self.lstm_state = state

        pitch = self.fcPitch(x)
        step = nn.LeakyReLU()(self.fcStep(x))
        duration = nn.LeakyReLU()(self.fcDuration(x))

        return pitch, step, duration

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def validation_epoch_end(self, _):
        generate_sample_song(self, filename=f"song_{self.val_epoch_num}.midi")

        self.val_epoch_num += 1

    def training_step(self, train_batch, batch_idx):
        input_notes, label_note = train_batch

        label_pitch = label_note[:, 0].long()
        val_step = label_note[:, 1]
        val_duration = label_note[:, 2]

        pitch, step, duration = self.forward(input_notes.float())

        step = torch.flatten(step)
        duration = torch.flatten(duration)

        loss_pitch = self.pitch_loss_function(pitch, label_pitch)
        loss_step = self.step_loss_function(step, val_step)
        loss_step_2 = self.step_loss_function_2(step)
        loss_duration = self.duration_loss_function(duration, val_duration)
        loss_duration_2 = self.duration_loss_function_2(duration)

        self.log('pitch_train_loss', loss_pitch, on_step=True, on_epoch=True)
        self.log('step_train_loss', loss_step, on_step=True, on_epoch=True)
        self.log('duration_train_loss', loss_duration, on_step=True, on_epoch=True)
        self.log('duration_max_train', torch.max(duration), on_step=True, on_epoch=True)
        self.log('duration_min_train', torch.min(duration), on_step=True, on_epoch=True)
        self.log('step_max_train', torch.max(step), on_step=True, on_epoch=True)
        self.log('step_min_train', torch.min(step), on_step=True, on_epoch=True)
        self.log('step_zero_loss_train', loss_step_2, on_step=True, on_epoch=True)
        self.log('duration_zero_loss_train', loss_duration_2, on_step=True, on_epoch=True)

        loss = loss_pitch + loss_step + loss_duration + loss_step_2 + loss_duration_2

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pitch_outputs = F.softmax(pitch, dim=1)

        acc = self.train_acc(pitch_outputs, label_pitch)
        self.log('train_pitch_acc', acc, on_epoch=True, on_step=False)

        f1 = self.train_macro_f1(pitch_outputs, label_pitch)
        self.log('train_pitch_macro_f1', f1, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_notes, label_note = val_batch

        label_pitch = label_note[:, 0].long()
        val_step = label_note[:, 1]
        val_duration = label_note[:, 2]

        pitch, step, duration = self.forward(input_notes.float())

        step = torch.flatten(step)
        duration = torch.flatten(duration)

        loss_pitch = self.pitch_loss_function(pitch, label_pitch)
        loss_step = self.step_loss_function(step, val_step)
        loss_step_2 = self.step_loss_function_2(step)
        loss_duration = self.duration_loss_function(duration, val_duration)
        loss_duration_2 = self.duration_loss_function_2(duration)

        self.log('pitch_train_loss', loss_pitch, on_step=True, on_epoch=True)
        self.log('step_train_loss', loss_step, on_step=True, on_epoch=True)
        self.log('duration_train_loss', loss_duration, on_step=True, on_epoch=True)
        self.log('duration_max_train', torch.max(duration), on_step=True, on_epoch=True)
        self.log('duration_min_train', torch.min(duration), on_step=True, on_epoch=True)
        self.log('step_max_train', torch.max(step), on_step=True, on_epoch=True)
        self.log('step_min_train', torch.min(step), on_step=True, on_epoch=True)
        self.log('step_zero_loss_train', loss_step_2, on_step=True, on_epoch=True)
        self.log('duration_zero_loss_train', loss_duration_2, on_step=True, on_epoch=True)

        loss = loss_pitch + loss_step + loss_duration + loss_step_2 + loss_duration_2

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        pitch_outputs = F.softmax(pitch, dim=1)

        acc = self.val_acc(pitch_outputs, label_pitch)
        self.log('val_pitch_acc', acc, on_epoch=True, on_step=False)

        f1 = self.val_macro_f1(pitch_outputs, label_pitch)
        self.log('val_pitch_macro_f1', f1, on_epoch=True, on_step=False)

        return loss


def generate_sample_song(model, song_len=200, seq_len=25, filename='sample_song.midi'):
    print(f"Generating a song: {filename}")
    song = get_random_song()
    song_dataset = SongsDataset([song], seq_len=seq_len, split_label=True)

    sample_song_notes = []
    x_song, _pred = song_dataset[0]

    for _ in range(0, song_len):
        x = torch.tensor([x_song])
        pitch, step, duration = model(x)

        pitch_class = np.argmax(pitch.numpy()[0])
        step = abs(step.numpy()[0][0])
        duration = abs(duration.numpy()[0][0])

        next_note = np.array([pitch_class, step, duration]).astype(np.float32)
        sample_song_notes.append(next_note)
        x_song = [*x_song[1:], next_note]

    notes = pd.DataFrame(sample_song_notes, columns=['pitch', 'step', 'duration'])
    notes_to_midi(notes, f"./lstm_songs/{filename}", INSTRUMENT_NAME)


if __name__ == '__main__':
    dm = SongsDataModule(num_train_songs=10, num_val_songs=5, split_label=True)
    model = SongLTSM(1)
    logger = TensorBoardLogger("lightning_logs", name="lstm_model")

    trainer = pl.Trainer(logger=logger, max_epochs=10, log_every_n_steps=1)
    trainer.fit(model, dm)
