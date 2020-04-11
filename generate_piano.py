# # AIMUSIC

# # Dependencies
print("[!]INFO: Importing dependencies ...")

# General
import numpy as np
import pandas as pd
from glob import glob
import IPython
import pickle

# Data preprocessing
from music21 import converter, instrument, note, chord, stream
import music21

# Model
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

# # Global Variables
print("[!]INFO: Setting up global variables ...")

DATASET = glob('data/dataset/piano/*.mid')
WEIGHTS_COUNTER = int(input("Enter model-weights file number: "))
NUM_OF_NOTES_TO_BE_GENERATED = int(input("Enter the number of notes to be generated: "))
GENERATED_MIDI = 'generated/piano/'
GENERATED_MIDI += input("Enter name of output file: ")
GENERATED_MIDI += '.mid'
INPUT_SEQUENCE_LENGTH = 100
CHECKPOINT_WEIGHTS = 'weights/weights.dataset.piano.'+ str(WEIGHTS_COUNTER) +'.hdf5'

print()

print("========================================================")
print("Model specification:")
print("--------------------")
print("Train Dataset: ./data/dataset/piano/*.mid")
print("Length of Train Dataset:", len(DATASET))
print("Number of Notes to be Generated:", NUM_OF_NOTES_TO_BE_GENERATED)
print("Input Sequence Length: ", INPUT_SEQUENCE_LENGTH)
print("Model-Weights File:", CHECKPOINT_WEIGHTS)
print("Output Directory:", GENERATED_MIDI)
print("========================================================")

print()


def init_network(network_in, n_vocab):
    print("\t[!]INFO: Initializing model.")

    try:
        model = Sequential()
        model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        print("\t[+]SUCCESS: Successfully initialized model.")
        return model
    except:
        print("\t[-]ERROR: Error in initializing model.")

# # Generating New MIDI files
def generate_midi():
    print("[!]INFO: Generating MIDI file in process ...")

    try:
        # Load the notes used to train the model
        with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
            notes = pickle.load(filepath)

        # Get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # Get all pitch names
        n_vocab = len(set(notes))
        network_input = get_input_sequences(notes, pitchnames, n_vocab)
        normalized_input = np.array(network_input)
        normalized_input = np.reshape(normalized_input, (len(network_input), INPUT_SEQUENCE_LENGTH, 1))

        model = init_network(normalized_input, n_vocab)
        print('\t[!]INFO: Loading model ...')
        model.load_weights(CHECKPOINT_WEIGHTS)
        print('\t[+]SUCCESS: Model loaded successfully.')

        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
        deextract_notes(prediction_output)

        print("\t[+]SUCCESS: Successfully generated MIDI file.")
    except:
        print("\t[-]ERROR: Error in generating MIDI file.")
    print("[!]INFO: MIDI file generation completed.\n")


def get_input_sequences(notes, pitchnames, n_vocab):
    # Map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    sequence_length = INPUT_SEQUENCE_LENGTH
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    return (network_input)


def generate_notes(model, network_input, pitchnames, n_vocab):
    # Pick a random integer
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # Pick a random sequence from the input as a starting point for the prediction
    pattern = network_input[start]

    with open('seed/seed', 'rb') as filepath:
        pattern = pickle.load(filepath)

    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction_output = []

    print('\t[!]INFO: Generating notes.')

    try:
        # Generate NUM_OF_NOTES_TO_BE_GENERATED notes
        for note_index in range(NUM_OF_NOTES_TO_BE_GENERATED):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            # Predicted output is the argmax(P(h|D))
            index = np.argmax(prediction)
            # Mapping the predicted interger back to the corresponding note
            result = int_to_note[index]
            # Storing the predicted output
            prediction_output.append(result)

            pattern.append(index)
            # Next input to the model
            pattern = pattern[1:len(pattern)]
        print("\t[+]SUCCESS: Successfully generated notes.")
        return prediction_output
    except:
        print("\t[-]ERROR: Error in generating notes.")


def deextract_notes(prediction_output):
    offset = 0
    output_notes = []

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Violin()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Violin()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=GENERATED_MIDI)

generate_midi()

## Ask user to enter unique notes for the first time
## Ask user to enter the length of notes so that audio may be generated of that length