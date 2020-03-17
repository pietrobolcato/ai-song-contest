import matplotlib.pyplot as plt
import math
import string
import warnings
import numpy as np
from tabulate import tabulate
from music21 import instrument, note, stream, duration

# remove future warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# the following makes the plot look nice
plot_params = {'legend.fontsize': 'x-large',
						 'figure.figsize': (15, 15),
						 'axes.labelsize': 'x-large',
						 'axes.titlesize':'x-large',
						 'xtick.labelsize':'x-large',
						 'ytick.labelsize':'x-large'}
plt.rcParams.update(plot_params)

# global parameters

params = {"syllable_midi_dir":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/lmd-full_and_reddit_MIDI_dataset/SyllableParsing/",
		  "euro_midi_dir":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/datasets/eurovision/",		  
		  "skip_gram_dir":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/datasets/skip-gram_lyric_encoders/",
		  "weights_dir":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/weights/clean/lstm_nt/",
		  "plot_dir":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/plot/clean/lstm_nt/",
		  "evaluation_dir":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/evaluation/",
		  "dump_features_file":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/dumps/features_nt_bd_19",
		  "euro_dump_features_file":"/home/pietrobolcato/Desktop/lstm/pp/masterthesis/pp_ubuntu/dumps/features_euro_19",
		  "sequence_length":19,
		  "min_frequency":10}

# common functions

"""
Description: Normalize the syllables under the same domain by applying lower case and strip, removing punctuation and numbers
Params:
  - syll (string): a syllable
Return:
  - the cleaned syllable
"""

def clean_syllable(syll):
	syll = syll.lower().strip() # lower and strip
	syll = syll.translate(str.maketrans('', '', string.punctuation)) # remove punctuation 
	syll = ''.join([i for i in s if not i.isdigit()]) # remove numbers

	return syll

"""
Description: Sample from a softmax distribution
Params:
  - preds (array): the softmax distribution
  - temperature (int): the temperature control the randomness of predictions (optional)
Return:
  - the sampled element
"""

def sample(preds, temperature=1.):
    EPSILON = 10e-16
    if temperature == 0.:
        return np.argmax(preds)

    preds = (np.asarray(preds)+EPSILON).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)

"""
Description: Convert MIDI number to note
Params:
  - number (int): the MIDI number
Return:
  - the corresponding note
"""

def number_to_note(number):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)

    octave = number // NOTES_IN_OCTAVE
    note = NOTES[number % NOTES_IN_OCTAVE]

    return note + " " + str(octave)

"""
Description: Convert note to MIDI number
Params:
  - note (string): the note
Return:
  - the corresponding MIDI number
"""

def note_to_number(midstr):
    Notes = [["C"],["C#","Db"],["D"],["D#","Eb"],["E"],["F"],["F#","Gb"],["G"],["G#","Ab"],["A"],["A#","Bb"],["B"]]
    answer = 0
    i = 0
    #Note
    letter = midstr[:-1]
    for note in Notes:
        for form in note:
            if letter.upper() == form:
                answer = i
                break
        i += 1
    #Octave
    answer += (int(midstr[-1]))*12
    return answer

"""
Description: Compute a summary with notes, durations and corresponding syllables
Params:
  - note (array): list of notes
  - duration (array): list of duration
  - syll (array): list of syllables
Return:
  - the summary
"""

def score(note, duration, syll):
	table = []
	for i in range(0,len(note)):
	  ntn = number_to_note(int(note[i]))
	  table.append([note[i], ntn, duration[i], syll[i]])

	tab = tabulate(table, headers=['Note (MIDI)', 'Note', 'Duration', 'Syllable'], tablefmt='orgtbl')
	return tab

"""
Description: Create a MIDI file given a sequence of notes and durations
Params:
  - pred_note (array): list of notes
  - pred_duration (array): list of duration
Return:
  - void
"""


def create_midi(pred_note, pred_duration, name='out.mid'):
	offset = 0
	output_notes = []

	for i in range(0,len(pred_note)):
		d = duration.Duration()
		d.quarterLength = pred_duration[i]

		new_note = note.Note(pred_note[i])
		new_note.duration = d
		new_note.offset = offset
		new_note.storedInstrument = instrument.SnareDrum()
		output_notes.append(new_note)

		offset += pred_duration[i]

	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp=name)