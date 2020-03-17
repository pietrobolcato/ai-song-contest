# utility to play all the bulk generated MIDI and select the most promising
# requires timidity installed in the computer

import subprocess
import glob

def play_midi(file, volume=800):
	bashCommand = "timidity --volume {} {}".format(volume, file)

	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

input_directory = "./Out_nt_longerseq"

midi_list = glob.glob('{}/*.mid'.format(input_directory))
save = []
decision = "3"
c = 0

for m in midi_list:
	print(m, "\t|\t", c, "/100")
	play_midi(m)
	decision = input("1: ok, 2: no, 3: again | ")
	while decision == "3":
		play_midi(m)
		decision = input("1: ok, 2: no, 3: again | ")

	if decision == "1":
		save.append(m)

	c += 1

f = open("selected.txt","a")
for s in save:
	f.write(s + "\n")

f.close()

print("Saved at: selected.txt")