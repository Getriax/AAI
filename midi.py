import tensorflow as tf
import pathlib
import numpy as np
import pretty_midi
import collections
import pandas as pd
import glob
import random

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  """
  :param midi_file: path to midi file
  :return: dataframe with notes { pitch, start, end, step, duration }
  """
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,
) -> pretty_midi.PrettyMIDI:
  """

  :param notes: list of notes { pitch, duration, step }[]
  :param out_file: name of the file to write to
  :param instrument_name: instrument to use
  :param velocity: note loudness
  :return: pretty midi instance with applied notes
  """

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

default_data_dir = 'data/maestro-v2.0.0'

def download_maestro_song_notes(data_path = default_data_dir):
  data_dir = pathlib.Path(data_path)

  if not data_dir.exists():
    tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.',
      cache_subdir='data',
    )

def get_maestro_song_notes(num_files = 5, skip = 0, data_path = default_data_dir):
  """
  :param num_files: number of files to scan
  :return: [[song1_notes],[song2_notes]]
  """
  data_dir = pathlib.Path(data_path)
  download_maestro_song_notes(data_path)

  filenames = glob.glob(str(data_dir / '**/*.mid*'))

  all_notes = []
  for f in filenames[skip:(num_files + skip)]:
    notes = midi_to_notes(f)
    all_notes.append(notes)

  return all_notes

def get_random_song(data_path = default_data_dir):
  data_dir = pathlib.Path(data_path)
  download_maestro_song_notes(data_path)

  filenames = glob.glob(str(data_dir / '**/*.mid*'))
  idx = random.randint(0, len(filenames) - 1)

  return midi_to_notes(filenames[idx])


