import os
import pickle
import numpy as np
import fractions

from tqdm import tqdm
from common import *
from music21 import converter
from glob import glob

"""
Description: Extract the features and creates the vocabulary from the LMD dataset directory and save them to disk
Params:
  - dir (string): the path to the LDM dataset
Return:
  - void
"""

def create_features_dump(dir):
	syllable_midi_files = os.listdir(dir) # get all the file inside the directory

	in_syll = [] # for the input syllable sequences
	in_notes = [] # for the input notes sequences
	in_duration = [] # for the input duration sequences

	syll_vocab = ["<xxx>"] 
	notes_vocab = [0]
	duration_vocab = [4]

	freq = {} 

	for f in tqdm(syllable_midi_files):
		current_file = params["syllable_midi_dir"] + f
		pairing = np.load(current_file, allow_pickle=True)[0] # load the midi-syllable pair

		i = 0
		current_seq_length = len(pairing[0])

		while (i + params["sequence_length"] + 1) < current_seq_length:
			current_syll = pairing[2][i:i + params["sequence_length"] + 1] # takes the current sequence of syllables
			current_midi = np.array(pairing[1][i: i + params["sequence_length"] + 1]) # takes the current sequence of midi
			current_note = current_midi[:, 0] # from the sequence of MIDI takes the note and the duration
			current_duration = current_midi[:, 1]

			# add sequence to the lists

			in_syll.append(current_syll)
			in_notes.append(current_note)
			in_duration.append(current_duration)

			# create vocab

			for s in current_syll:
				s = clean_syllable(s)

				if s not in freq: # count the frequency of every syllable
					freq[s] = 0
				else:
					freq[s] += 1

				if freq[s] >= params["min_frequency"] and s not in syll_vocab:
					syll_vocab.append(s)

			for n in current_note:
				if n not in notes_vocab and n >= 24 and n < 107: # keep only the pitches from c1 to b7
					notes_vocab.append(n)

			for d in current_duration:
				if d not in duration_vocab and d <= 4: # keep only the durations between 0.25 and 4.0
					duration_vocab.append(d)

			i = i + params["sequence_length"] + 1

	# define index maps

	syll_idx = {val:index for index, val in enumerate(syll_vocab)}
	idx_syll = {index:val for index, val in enumerate(syll_vocab)}

	notes_idx = {val:index for index, val in enumerate(notes_vocab)}
	idx_notes = {index:val for index, val in enumerate(notes_vocab)}

	duration_idx = {val:index for index, val in enumerate(duration_vocab)}
	idx_duration = {index:val for index, val in enumerate(duration_vocab)}

	# create encoded sequences

	for i in range(len(in_syll)):
		for j in range(params["sequence_length"] + 1): # for every sequence
			current_syll = clean_syllable(in_syll[i][j])
			current_note = in_notes[i][j]
			current_duration = in_duration[i][j]

			if current_syll not in syll_vocab: # if not in the vocab, add the unknown token
				in_syll[i][j] = syll_idx["<xxx>"]
			else: # otherwise encode it
				in_syll[i][j] = syll_idx[current_syll]

			if current_note not in notes_vocab:
				in_notes[i][j] = notes_idx[0]
			else:
				in_notes[i][j] = notes_idx[current_note]

			if current_duration not in duration_vocab:
				in_duration[i][j] = duration_idx[4]
			else:
				in_duration[i][j] = duration_idx[current_duration]

	# save the extracted features to disk

	unified = {"in_syll":in_syll,
			   "in_notes":in_notes,
			   "in_duration":in_duration,
			   "syll_vocab":syll_vocab,
			   "notes_vocab":notes_vocab,
			   "duration_vocab":duration_vocab}

	dump_open = open(params["dump_features_file"], 'wb')
	pickle.dump(unified, dump_open)
	dump_open.close()

	print("Saved to: ", params["dump_features_file"])

"""
Description: Extract the features from the Eurovision dataset directory and save them to disk
Params:
  - dir (string): the path to the Eurovision dataset
Return:
  - void
"""

def create_euro_features_dump(dir): 
	# first load the vocabulary from the LMD dataset

	if not os.path.exists(params["dump_features_file"]):
		create_features_dump(params["syllable_midi_dir"])

	dataset = pickle.load( open(params["dump_features_file"], 'rb') )	
	syll_vocab = dataset["syll_vocab"]
	notes_vocab = dataset["notes_vocab"]
	duration_vocab = dataset["duration_vocab"]

	# parse all the midi files and add notes and durations to a list		

	pre_notes = []
	pre_duration = []

	for file in tqdm(glob("{}/*.midi".format(dir))):
		midi = converter.parse(file)

		notes_to_parse = None

		try: # file has instrument parts
			s2 = instrument.partitionByInstrument(midi)
			notes_to_parse = s2.parts[0].recurse() 
		except: # file has notes in a flat structure
			notes_to_parse = midi.flat.notes

		for element in notes_to_parse:
			if isinstance(element, note.Note):
				midi_note = note_to_number(str(element.pitch))
				pre_notes.append(midi_note)

				duration = element.duration.quarterLength
				if isinstance(duration, fractions.Fraction):
					pre_duration.append(0.25)
				else:
					pre_duration.append(duration)

	# define index maps

	notes_idx = {val:index for index, val in enumerate(notes_vocab)}
	idx_notes = {index:val for index, val in enumerate(notes_vocab)}

	duration_idx = {val:index for index, val in enumerate(duration_vocab)}
	idx_duration = {index:val for index, val in enumerate(duration_vocab)}

	in_notes = []
	in_duration = []
	in_syll = []

	# divide in the right sequence length

	i = 0
	while (i + params["sequence_length"] + 1) < len(pre_notes):
		current_syll = np.zeros( (params["sequence_length"] + 1,), dtype=int) # no syll in the eurovision dataset
		current_note = np.array(pre_notes[i: i + params["sequence_length"] + 1])
		current_duration = np.array(pre_duration[i: i + params["sequence_length"] + 1])

		in_syll.append(current_syll)
		in_notes.append(current_note)
		in_duration.append(current_duration)

		i = i + params["sequence_length"] + 1

	# encode

	for i in range(0,len(in_notes)):
		for j in range(0,len(in_notes[i])):
			current_note = in_notes[i][j]
			current_duration = in_duration[i][j]
			
			if current_note not in notes_vocab:
				in_notes[i][j] = notes_idx[0]
			else:
				in_notes[i][j] = notes_idx[current_note]

			if current_duration not in duration_vocab:
				in_duration[i][j] = duration_idx[4]
			else:
				in_duration[i][j] = duration_idx[current_duration]


	# save the extracted features to disk

	unified = { "in_syll":in_syll,
				"in_notes":in_notes,
				"in_duration":in_duration,
				"syll_vocab":syll_vocab,
				"notes_vocab":notes_vocab,
				"duration_vocab":duration_vocab}

	dump_open = open(params["euro_dump_features_file"], 'wb')
	pickle.dump(unified, dump_open)
	dump_open.close()

	print("Saved to: ", params["euro_dump_features_file"])