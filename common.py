import numpy as np
import pandas as pd
import torch
from torch import nn

from constants import INSTRUMENT_NAME
from midi import get_random_song, notes_to_midi
from songs_data import SongsDataset


class ZeroLoss(nn.Module):
    '''
    Funkcja straty która karze za wartosci=0
    '''

    def __init__(self, scale=False):
        self.scale = scale
        super(ZeroLoss, self).__init__()

    def forward(self, inputs):
        zeros = inputs[inputs == 0]

        if self.scale:
            return len(zeros) / len(inputs)

        return len(zeros)


def generate_sample_song(model, song_len=200, seq_len=24, filename='sample_song.midi', has_state=False):
    '''
    Generuje przykładową piosenkę, z wprowadzona liczba nut, wykorzystujac przekazany model.
    :param model: model generujacy
    :param song_len: dlugosc piosenki
    :param seq_len: ilosc nut w sekwencji
    :param filename: nazwa pliku
    :param has_state: czy istnieje dodatkowy parametr stanu
    :return:
    '''
    print(f"Generating a song: {filename}")
    song = get_random_song()
    song_dataset = SongsDataset([song], seq_len=seq_len)

    state = None if not has_state else model.init_hidden(1)

    sample_song_notes = []
    x_song = song_dataset[0]

    for _ in range(0, song_len):
        x = torch.tensor([x_song])

        if has_state:
            pitch, step, duration, state = model(x, state)
        else:
            pitch, step, duration = model(x)

        pitch_class = pitch[0].argmax().item()
        step = abs(step[0].item())
        duration = abs(duration[0].item())

        next_note = np.array([pitch_class, step, duration]).astype(np.float32)
        sample_song_notes.append(next_note)
        x_song = [*x_song[1:], next_note]

    notes = pd.DataFrame(sample_song_notes, columns=['pitch', 'step', 'duration'])
    notes_to_midi(notes, f"./gan_songs/{filename}", INSTRUMENT_NAME)
