from common import *
from preprocess import *
from neural_network import *
from evaluation import *

import os
import argparse

# extract the features if not already done 

if not os.path.exists(params["dump_features_file"]):
	create_features_dump(params["syllable_midi_dir"])

if not os.path.exists(params["euro_dump_features_file"]):
	create_euro_features_dump(params["euro_midi_dir"])

# define argparser

parser = argparse.ArgumentParser("")
subparser = parser.add_subparsers(dest="command")

parser.add_argument('--eurovision', action='store_true', help="specify whether using the Eurovision dataset or not")

train = subparser.add_parser("train", help="train the network")
train.add_argument("run_name", help="the name of the run. The weights and plot file will be name accordingly")
train.add_argument("epochs", type=int, default=100, nargs="?", help="number of epochs")
train.add_argument("load", type=bool, default=False, nargs="?", help="boolean value to indicate whether to load existing weights or not")
train.add_argument("append_name", default="", nargs="?", help="append a name to the end of 'run_name'. Useful in case one wants to load weights but save them under a different name")
train.add_argument("batch_size", type=int, default=128, nargs="?", help="dimension of the training batch")

generate_bulk = subparser.add_parser("generate_bulk", help="generate in bulk MIDI and lyrics from the trained network")
generate_bulk.add_argument("run_name", help="name of the run in order to load the weights")
generate_bulk.add_argument("quantity", type=int, default=100, help="number of MIDI files to generate")
generate_bulk.add_argument("out_directory", help="output directory for the generated MIDI files")

evaluation_bl = subparser.add_parser("evaluation_baseline", help="generate evaluation plots and statistics for the dataset")
evaluation_bl.add_argument("name_out", help="name of the evaluation run")
evaluation_bl.add_argument("ngram", type=int, default=5, nargs="?", help="number of ngrams to search for")
evaluation_bl.add_argument("show", type=bool, default=False, nargs="?", help="boolean value to indicate whether the graphs will be shown through a GUI or not")

evaluation_gen = subparser.add_parser("evaluation_generated", help="generate evaluation plot and statistics for the bulk generated MIDI files")
evaluation_gen.add_argument("run_name", help="name of the run in order to load the weights")
evaluation_gen.add_argument("name_out", help="name of the evaluation run")
evaluation_gen.add_argument("quantity", type=int, default=500, nargs="?", help="number of MIDI files to generate")
evaluation_gen.add_argument("ngram", type=int, default=5, nargs="?", help="number of ngrams to search for")
evaluation_gen.add_argument("show", type=bool, default=False, nargs="?", help="boolean value to indicate whether the graphs will be shown through a GUI or not")

cmd = parser.parse_args()

# load either the eurovision or the LMD dataset

if cmd.eurovision:
	print("Using eurovision dataset")
	features_file = params["euro_dump_features_file"] 
	loss_weights = [1,1,0]
else:
	print("Using LMD dataset")
	features_file = params["dump_features_file"]
	loss_weights = [1,1,1]

dataset = pickle.load( open(features_file, 'rb') )	

in_syll = dataset["in_syll"]
in_notes = dataset["in_notes"]
in_duration = dataset["in_duration"]
syll_vocab = dataset["syll_vocab"]
notes_vocab = dataset["notes_vocab"]
duration_vocab = dataset["duration_vocab"]

syll_vocab_size = len(syll_vocab)
notes_vocab_size = len(notes_vocab)
duration_vocab_size = len(duration_vocab)

# init the neural network

nn = NN(in_notes, in_duration, in_syll, notes_vocab_size, duration_vocab_size, syll_vocab_size, notes_vocab, duration_vocab, syll_vocab, loss_weights=loss_weights)

# parse the arg command

if cmd.command == "train":
	nn.train(cmd.run_name, cmd.epochs, load=cmd.load, append_name=cmd.append_name, batch_size=cmd.batch_size)

elif cmd.command == "generate_bulk":
	nn.generate_bulk(cmd.quantity, cmd.run_name, cmd.out_directory)

elif cmd.command == "evaluation_baseline":
	evaluation_baseline = Evaluation(in_notes, in_duration, notes_vocab, duration_vocab, cmd.name_out)
	evaluation_baseline.complete_report(ngram = cmd.ngram, show = cmd.show)		

elif cmd.command == "evaluation_generated":
	eval_notes = []
	eval_duration = []

	for i in tqdm(range(cmd.quantity)):
		p = nn.generate_sequence(cmd.run_name, verbose=False)
		eval_notes.append(p[0])
		eval_duration.append(p[1])

	evaluation_gen = Evaluation(eval_notes, eval_duration, notes_vocab, duration_vocab, cmd.name_out)
	evaluation_gen.complete_report(ngram = cmd.ngram, show = cmd.show)