import pickle

# # Helper Functions

def get_int_to_note(WEIGHTS_COUNTER):
    # Load the notes used to train the model
    with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    return dict((number, note) for number, note in enumerate(pitchnames))


def get_note_to_int(WEIGHTS_COUNTER):
    # Load the notes used to train the model
    with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    return dict((note, number) for number, note in enumerate(pitchnames))


def get_vocabulary(WEIGHTS_COUNTER):
    # Load the notes used to train the model
    with open('data/notes/notes_'+str(WEIGHTS_COUNTER), 'rb') as filepath:
        notes = pickle.load(filepath)

    return sorted(set(item for item in notes))