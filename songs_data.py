import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split
import pytorch_lightning as pl
import torch

from constants import BATCH_SIZE, INSTRUMENT_NAME
from midi import download_maestro_song_notes, get_maestro_song_notes, get_random_song, notes_to_midi

NUM_WORKERS = int(os.cpu_count() / 2)

class SongsDataset(Dataset):
    def __init__(self, songs, seq_len, split_label=False):
        self.seq_len = seq_len
        self.songs = songs
        self.song_idx = 0
        self.songs_switch_idx = 0
        self.note_fields = ["pitch", "step", "duration"]
        self.split_label =split_label

    def __len__(self):
        total = sum(map(len, self.songs)) - ((self.seq_len + 1) * len(self.songs))
        return total - (total % self.seq_len)

    def __getitem__(self, idx):
        notes = self.songs[self.song_idx]
        notes_start_idx = idx - self.songs_switch_idx

        # when current song notes length is less than required move to the next song
        if notes_start_idx + self.seq_len >= len(notes):
            self.song_idx += 1
            self.songs_switch_idx = idx
            notes_start_idx = 0

            notes = self.songs[self.song_idx]

        x = notes[notes_start_idx:(notes_start_idx + self.seq_len)]
        x = x[self.note_fields].astype(np.float32).values

        if (self.split_label):
            return x[0:self.seq_len-1], x[self.seq_len-1]

        return x


class SongsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = BATCH_SIZE,
            num_workers: int = NUM_WORKERS,
            seq_len=25,
            num_train_songs: int = 2,
            num_val_songs: int = 1,
            num_test_songs: int = 1,
            split_label=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train_songs = num_train_songs
        self.num_val_songs = num_val_songs
        self.num_test_songs = num_test_songs
        self.seq_len = seq_len
        self.split_label = split_label

    def prepare_data(self):
        # download
        download_maestro_song_notes()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            songs = get_maestro_song_notes(self.num_train_songs + self.num_val_songs)
            self.songs_train, self.songs_val = random_split(songs, [self.num_train_songs, self.num_val_songs])

            self.songs_train = SongsDataset(self.songs_train, self.seq_len, split_label=self.split_label)
            self.songs_val = SongsDataset(self.songs_val, self.seq_len, split_label=self.split_label)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.songs_test = get_maestro_song_notes(self.num_test_songs,
                                                     skip=self.num_train_songs + self.num_val_songs)
            self.songs_test = SongsDataset(self.songs_test, self.seq_len, split_label=self.split_label)

    def train_dataloader(self):
        return DataLoader(
            self.songs_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.songs_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.songs_test, batch_size=self.batch_size, num_workers=self.num_workers)


