import pickle
import random

from helper import *

def generate_random_initial_sequence(INPUT_SEQUENCE_LENGTH):
    WEIGHTS_COUNTER = int(input("Enter weight file number to generate the seed: "))

    print("[!]INFO: Generating random seed notes.")

    try:
        note_to_int = get_note_to_int(WEIGHTS_COUNTER)
        int_to_note = get_int_to_note(WEIGHTS_COUNTER)

        inputSequence = list()
        for i in range(INPUT_SEQUENCE_LENGTH):
            inputSequence.append(note_to_int[int_to_note[random.randint(0, len(int_to_note)-1)]])

        with open('seed/seed_'+str(WEIGHTS_COUNTER), 'wb') as filepath:
            pickle.dump(inputSequence, filepath)
        print("[+]SUCCESS: Seed notes generated successfully.")
    except:
        print("[-]ERROR: Error in generating seed notes.")
    # return inputSequence
generate_random_initial_sequence(100)
