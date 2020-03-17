import collections
import matplotlib.pyplot as plt
import numpy as np

from common import *
from itertools import tee, islice
from collections import Counter

class Evaluation:
	"""
	Description: Initialize the evaluation class
	Params:
	- notes (array): array of preprocessed input notes
	- duration (array): array of preprocessed input durations
	- notes_vocab (array): notes vocabulary
	- duration_vocab (array): durations vocabulary
	- report_name (string): name of the evaluation report
	Return:
	- void
	"""

	def __init__(self, notes, duration, notes_vocab, duration_vocab, report_name):
		self.notes = notes
		self.duration = duration
		self.report_name = report_name
		self.notes_vocab = notes_vocab
		self.duration_vocab = duration_vocab

		self.notes_idx = {val:index for index, val in enumerate(self.notes_vocab)}
		self.idx_notes = {index:val for index, val in enumerate(self.notes_vocab)}

		self.duration_idx = {val:index for index, val in enumerate(self.duration_vocab)}
		self.idx_duration = {index:val for index, val in enumerate(self.duration_vocab)}

	"""
	Description: Auxiliary function to count the freq of items in an array
	Params:
	- array (array): input array
	Return:
	- dictionary with the frequency of every element in the array
	"""

	def count_freq(self, array):
		c = collections.Counter(array)
		c = sorted(c.items())
		name = [i[0] for i in c]
		freq = [i[1] for i in c]

		return name, freq

	"""
	Description: Auxiliary function to count the number of n-grams
	Params:
	- lst (array): input array
	- n (int): dimensionality of the n-gram
	Return:
	- void
	"""

	def count_ngrams(self, lst, n):
		tlst = lst
		while True:
			a, b = tee(tlst)
			l = tuple(islice(a, n))
			if len(l) == n:
				yield l
				next(b)
				tlst = b
			else:
				break

	"""
	Description: Calculate the pitch distribution
	Params:
	- show (bool): boolean value to indicate whether the graphs will be shown through a GUI or not
	- avg (bool): whether to calculate the statistics on the average of the items or not
	Return:
	- the pitch distribution
	"""

	def pitch_distribution(self, show = True, avg = True):
		notes_all = []

		for song in self.notes:
			for n in song:
				notes_all.append(self.idx_notes[n])

		notes_name, notes_freq = self.count_freq(notes_all)
		if avg:
			notes_freq = np.array(notes_freq) / len(self.notes)

		plt.bar(notes_name, notes_freq)
		plt.xlabel("Pitch")
		plt.ylabel("Number of occurrence")
		plt.title("{} | Distribution of pitches".format(self.report_name))
		plt.xticks(np.arange(0, np.max(notes_all), 3.0))
		plt.savefig("{}pitch_distribution_{}.png".format(params["evaluation_dir"], self.report_name))
		if show:
			plt.show()

		plt.close()

		return notes_name, notes_freq

	"""
	Description: Calculate the duration distribution
	Params:
	- show (bool): boolean value to indicate whether the graphs will be shown through a GUI or not
	- avg (bool): whether to calculate the statistics on the average of the items or not
	Return:
	- the duration distribution
	"""

	def duration_distribution(self, show = True, avg = True):
		duration_all = []

		for song in self.duration:
			for d in song:
				duration_all.append(self.idx_duration[d])

		duration_name, duration_freq = self.count_freq(duration_all)
		if avg:
			duration_freq = np.array(duration_freq) / len(self.notes)

		plt.bar(duration_name, duration_freq, width=0.2)
		plt.xlabel("Duration")
		plt.ylabel("Number of occurrence")
		plt.title("{} | Distribution of durations".format(self.report_name))
		plt.xticks(self.duration_vocab)
		plt.savefig("{}duration_distribution_{}.png".format(params["evaluation_dir"], self.report_name))
		if show:
			plt.show()

		plt.close()

		return duration_name, duration_freq

	"""
	Description: Calculate the interval distribution
	Params:
	- show (bool): boolean value to indicate whether the graphs will be shown through a GUI or not
	- avg (bool): whether to calculate the statistics on the average of the items or not
	Return:
	- the interval distribution
	"""

	def interval_distribution(self, show = True, avg = True):
		freq = {}

		for song in self.notes:
			for n in range(0,len(song)-1):
				interval = self.idx_notes[song[n]] - self.idx_notes[song[n+1]]

				if interval >= 0 and interval <= 11:
					if interval not in freq:
						freq[interval] = 0
					else:
						freq[interval] += 1
		
		sort = sorted(freq.items())
		int_name = [i[0] for i in sort]
		int_freq = [i[1] for i in sort]

		if avg:
			int_freq = np.array(int_freq) / len(self.notes)

		plt.bar(int_name, int_freq)
		plt.xlabel("Interval")
		plt.ylabel("Number of occurrence")
		plt.title("{} | Distribution of intervals".format(self.report_name))
		plt.xticks(np.arange(0, 11, 1.0))
		plt.savefig("{}interval_distribution_{}.png".format(params["evaluation_dir"], self.report_name))
		if show:
			plt.show()

		plt.close()

		return int_name, int_freq

	"""
	Description: Calculate the n-grams distribution
	Params:
	- ngram (int): maximum dimensionality of the n-gram
	- show (bool): boolean value to indicate whether the graphs will be shown through a GUI or not
	- avg (bool): whether to calculate the statistics on the average of the items or not
	Return:
	- the n-grams distribution
	"""

	def ngram_distribution(self, ngram, show = True, avg = True): 
		notes_all = []
		ngram_freq = np.zeros(ngram-1)

		for song in self.notes:
			for n in range(2, ngram+1):
				temp_c = 0
				c = Counter(self.count_ngrams(song, n))
				for i in c:
					val = c[i]
					if val > 1:
						temp_c += 1

				ngram_freq[n-2] += temp_c

		ngram_name = np.arange(2,ngram+1,1)
		if avg:
			ngram_freq /= len(self.notes)

		plt.bar(ngram_name, ngram_freq)
		plt.xlabel("N-gram")
		plt.ylabel("Number of occurrence")
		plt.xticks(ngram_name)
		plt.title("{} | Distribution of N-gram".format(self.report_name))
		plt.savefig("{}ngram_distribution_{}.png".format(params["evaluation_dir"], self.report_name))
		if show:
			plt.show()

		plt.close()

		return ngram_name, ngram_freq

	"""
	Description: Calculate the MIDI number span, unique MIDI number, song length 
	Params:
	- show (bool): boolean value to indicate whether the statistics will be printed or not
	- avg (bool): whether to calculate the statistics on the average of the items or not
	Return:
	- the calculated statistics
	"""

	def numerical_statistics(self, show = True, avg = True): 
		stats = {"midi_span":0,
				 "midi_unique":0,
				 "song_length":0}

		for song in self.notes:
			span = abs(max(song)-min(song))
			unique = len(set(song))

			stats["midi_span"] += span
			stats["midi_unique"] += unique

		for song in self.duration:
			temp_length = 0
			for d in song:
				temp_length += d

			stats["song_length"] += temp_length

		if avg:
			stats["midi_span"] /= len(self.notes)
			stats["midi_unique"] /= len(self.notes)
			stats["song_length"] /= len(self.notes)

		vals = [ v for v in stats.values() ]
		tab = tabulate([vals], headers=["MIDI Numbers span", "Number of unique MIDI Numbers", "Song length"], tablefmt='orgtbl')
		with open("{}statistics_{}.txt".format(params["evaluation_dir"], self.report_name), "w") as stat_file:
			stat_file.write(tab)

		if show:
			print(tab)
		
		return stats

	"""
	Description: Create a complete report using all the metrics defined above
	Params:
	- ngram (int): maximum dimensionality of the n-gram
	- show (bool): boolean value to indicate whether the statistics will be printed or not
	- avg (bool): whether to calculate the statistics on the average of the items or not
	Return:
	- void
	"""

	def complete_report(self, ngram, show = False, avg = True):
		self.pitch_distribution(show=show, avg=avg)
		self.duration_distribution(show=show, avg=avg)
		self.interval_distribution(show=show, avg=avg)
		self.ngram_distribution(ngram, show=show, avg=avg)
		self.numerical_statistics(show=show, avg=avg)