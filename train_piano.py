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
EPOCHS = int(input("Enter the number of epochs to train for: "))

INPUT_SEQUENCE_LENGTH = 100
CHECKPOINT_WEIGHTS = 'weights/weights.dataset.piano.'+ str(WEIGHTS_COUNTER) +'.hdf5'

print()

print("========================================================")
print("Model specification:")
print("--------------------")
print("Train Dataset: ./data/dataset/piano/*.mid")
print("Length of Train Dataset:", len(DATASET))
print("Epochs:", EPOCHS)
print("Input Sequence Length: ", INPUT_SEQUENCE_LENGTH)
print("Model-Weights File:", CHECKPOINT_WEIGHTS)
print("========================================================")

print()

# # Extracting Notes
def extract_notes():
    print("\t[!]INFO: Extracting notes from dataset.")

    notes = []
    try:
        for file in DATASET:
            # Converting .mid file to stream object/bytes
            midi = converter.parse(file)
            notes_to_parse = []

            parts = None
            try:
                # Given a single stream, partition into a part for each unique instrument
                parts = instrument.partitionByInstrument(midi)
            except:
                pass

            if parts != None:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse: 
                if isinstance(element, note.Note):
                    # If element is a note, extract pitch
                    notes.append(str(element.pitch))
                elif(isinstance(element, chord.Chord)):
                    # If element is a chord, append the normal form of the chord (a list of integers) to the list of notes
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        print("\t[+]SUCCESS: Successfully extracted notes.")
    except:
        print("\t[-]ERROR: Error in extracting notes.")
    
    # Saving the notes to a file
    print("\t[!]INFO: Saving notes to file.")
    
    try:
        with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'wb') as filepath:
            pickle.dump(notes, filepath)
        print("\t[+]SUCCESS: Successfully written notes to file.")
        return notes
    except:
        print("\t[-]ERROR: Error in writing notes to file.")


# # Preparing Vocabulary
def prepare_sequence_vocab(notes, n_vocab): 
    print("\t[!]INFO: Preparing input sequences/vocabulary.")
    
    try:
        sequence_length = INPUT_SEQUENCE_LENGTH

        # Extract the unique pitches in the list of notes.
        pitchnames = sorted(set(item for item in notes))

        # Create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        network_input = []
        network_output = []

        # Create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i: i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # Reshape the input into a format comatible with LSTM layers 
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

        # Normalize input
        network_input = network_input / float(n_vocab)

        # One hot encode the output vectors
        network_output = np_utils.to_categorical(network_output)

        print("\t[+]SUCCESS: Successfully prepared vocabulary.")
        return network_input, network_output
    except:
        print("\t[-]ERROR: Error in preparing vocabulary.")


# # Initializing Network
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


# # Model Training
def train(model, network_input, network_output, epochs):
    print("\t[!]INFO: Model training in process.")
    
    try:
        # Create checkpoint to save the best model weights.
        checkpoint = ModelCheckpoint(CHECKPOINT_WEIGHTS, monitor='loss', verbose=0, save_best_only=True)

        model.fit(network_input, network_output, epochs=epochs, batch_size=32, callbacks=[checkpoint])
        print("\t[+]SUCCESS: Successfully trained the model.")
    except:
        print("\t[-]ERROR: Error in training the model.")

def train_network():
    print("[!]INFO: Data-preprocessing in process ...")
    notes = extract_notes()
    n_vocab = len(set(notes))
    network_in, network_out = prepare_sequence_vocab(notes, n_vocab)
    print("[!]INFO: Data-preprocessing completed.\n")
    
    print("[!]INFO: Model training in process ...")
    model = init_network(network_in, n_vocab)
    train(model, network_in, network_out, EPOCHS)
    print("[!]INFO: Model training completed.\n")
    return model

train_network()

# # Helper Functions

def get_int_to_note():
    #Load the notes used to train the model
    with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))    
    return dict((number, note) for number, note in enumerate(pitchnames))

def get_note_to_int():
    #Load the notes used to train the model
    with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))    
    return dict((note, number) for number, note in enumerate(pitchnames))
    
def get_vocabulary():
    #Load the notes used to train the model
    with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
        notes = pickle.load(filepath)

    return sorted(set(item for item in notes))

## Ask user to enter unique notes for the first time
## Ask user to enter the length of notes so that audio may be generated of that length

