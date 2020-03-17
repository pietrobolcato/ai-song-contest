from common import *
from tqdm import tqdm
from keras import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, History
from keras.layers import Input, LSTM, TimeDistributed, Dense, concatenate

class NN:
	"""
	Description: Initialize the neural network class and create the model
	Params:
	- in_notes_enc (array): array of preprocessed input notes
	- in_duration_enc (array): array of preprocessed input durations
	- in_syll_enc (array): array of preprocessed input syllables
	- notes_vocab_size (int): dimension of the notes vocabulary
	- duration_vocab_size (int): dimension of the durations vocabulary
	- syll_vocab_size (int): dimension of the syllables vocabulary
	- notes_vocab (array): notes vocabulary
	- duration_vocab (array): durations vocabulary
	- syll_vocab (array): syllables vocabulary
	- loss_weights (list): weights for the three loss functions of the neural network (optional)
	Return:
	- void
	"""

	def __init__(self, in_notes_enc, in_duration_enc, in_syll_enc, notes_vocab_size, duration_vocab_size, syll_vocab_size, notes_vocab, duration_vocab, syll_vocab, loss_weights=[1,1,1]):
		self.in_notes_enc = in_notes_enc
		self.in_duration_enc = in_duration_enc
		self.in_syll_enc = in_syll_enc
		self.notes_vocab_size = notes_vocab_size
		self.duration_vocab_size = duration_vocab_size
		self.syll_vocab_size = syll_vocab_size
		self.notes_vocab = notes_vocab
		self.duration_vocab = duration_vocab
		self.syll_vocab = syll_vocab

		self.model = self.create_model(loss_weights=loss_weights)

	"""
	Description: Create the neural network model
	Params:
	- loss_weights (list): weights for the three loss functions of the neural network (optional)
	Return:
	- the compiled model
	"""

	def create_model(self, loss_weights=[1,1,1]):
		note_input = Input(shape=(None,self.notes_vocab_size), name='note')
		note_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='note_features_1')(note_input)
		note_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='note_features_2')(note_features)
		note_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='note_features_3')(note_features)
		#note_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='note_features_4')(note_features)
		note_out = TimeDistributed(Dense(self.notes_vocab_size, activation='softmax'), name='note_out')(note_features)

		duration_input = Input(shape=(None,self.duration_vocab_size), name='duration')

		x_1 = concatenate([note_input, note_out, duration_input])
		duration_features = LSTM(256, return_sequences=True, recurrent_dropout=0.3, name='duration_features_1')(x_1)
		duration_features = LSTM(256, return_sequences=True, recurrent_dropout=0.3, name='duration_features_2')(duration_features)
		duration_features = LSTM(256, return_sequences=True, recurrent_dropout=0.3, name='duration_features_3')(duration_features)
		#duration_features = LSTM(256, return_sequences=True, recurrent_dropout=0.3, name='duration_features_4')(duration_features)
		duration_out = TimeDistributed(Dense(self.duration_vocab_size, activation='softmax'), name='duration_out')(duration_features)

		syll_input = Input(shape=(None, self.syll_vocab_size), name='syll')

		x_2 = concatenate([note_input, note_out, duration_input, duration_out, syll_input])
		syll_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='syll_features_1')(x_2)
		syll_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='syll_features_2')(syll_features)
		syll_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='syll_features_3')(syll_features)
		#syll_features = LSTM(512, return_sequences=True, recurrent_dropout=0.3, name='syll_features_4')(syll_features)
		syll_out = TimeDistributed(Dense(self.syll_vocab_size, activation='softmax'), name='syll_out')(syll_features)

		model = Model(inputs=[note_input, duration_input, syll_input],
				outputs=[note_out, duration_out, syll_out])

		model.compile(optimizer='adam',
						loss={'note_out': 'categorical_crossentropy',
								'duration_out': 'categorical_crossentropy',
								'syll_out': 'categorical_crossentropy'},
						loss_weights=loss_weights)
		return model

	"""
	Description: Define the generator to train the neural network
	Params:
	- batch_size (int): dimension of the training batch (optional)
	Return:
	- void
	"""

	def train_generator(self, batch_size = 128):
		note_x = []
		note_y = []
		duration_x = []
		duration_y = []
		syll_x = []
		syll_y = []

		c = 0

		while True:
			i = np.random.randint(0, len(self.in_notes_enc)) # select random sequence

			note_in = self.in_notes_enc[i][0:params["sequence_length"]]
			note_out = self.in_notes_enc[i][1:params["sequence_length"] + 1]
			note_x.append(to_categorical([note_in], num_classes=self.notes_vocab_size))
			note_y.append(to_categorical([note_out], num_classes=self.notes_vocab_size))

			duration_in = self.in_duration_enc[i][0:params["sequence_length"]]
			duration_out = self.in_duration_enc[i][1:params["sequence_length"] + 1]
			duration_x.append(to_categorical([duration_in], num_classes=self.duration_vocab_size))
			duration_y.append(to_categorical([duration_out], num_classes=self.duration_vocab_size))

			syll_in = self.in_syll_enc[i][0:params["sequence_length"]]
			syll_out = self.in_syll_enc[i][1:params["sequence_length"] + 1]
			syll_x.append(to_categorical([syll_in], num_classes=self.syll_vocab_size))
			syll_y.append(to_categorical([syll_out], num_classes=self.syll_vocab_size))

			c += 1

			if c == batch_size:
				note_x = np.array(note_x).reshape(batch_size,params["sequence_length"],self.notes_vocab_size)
				note_y = np.array(note_y).reshape(batch_size,params["sequence_length"],self.notes_vocab_size)

				duration_x = np.array(duration_x).reshape(batch_size,params["sequence_length"],self.duration_vocab_size)
				duration_y = np.array(duration_y).reshape(batch_size,params["sequence_length"],self.duration_vocab_size)

				syll_x = np.array(syll_x).reshape(batch_size,params["sequence_length"],self.syll_vocab_size)
				syll_y = np.array(syll_y).reshape(batch_size,params["sequence_length"],self.syll_vocab_size)

				dc = [[note_x, duration_x, syll_x], [note_y, duration_y, syll_y]]
				yield(dc)

				note_x = []
				note_y = []
				duration_x = []
				duration_y = []
				syll_x = []
				syll_y = []

				c = 0

	"""
	Description: Train the neural network and plot the graph of the losses over time
	Params:
	- name (string): name of the run. The weights and plot file will be name accordingly
	- epochs (int): number of epochs
	- load (bool): boolean value to indicate whether to load existing weights or not
	- append_name (string): append a name to the end of 'name'. Useful in case one wants to load weights but save them under a different name
	- batch_size (int): dimension of the training batch (optional)
	Return:
	- void
	"""

	def train(self, name, epochs, load = False, append_name = "", batch_size = 128):
		steps = len(self.in_notes_enc)/batch_size
		weights_file_path = "{}{}".format(params["weights_dir"],name)

		checkpoint = ModelCheckpoint(weights_file_path + append_name, monitor='val_loss', verbose=1, save_weights_only = True, period=1)

		if load == True:
			self.model.load_weights(weights_file_path)

		gen_train = self.train_generator(batch_size)
		history = self.model.fit_generator(gen_train, steps_per_epoch=steps, 
												 verbose=1,
												 epochs=epochs,
												 callbacks=[checkpoint])
		 
		plt.plot(history.history['note_out_loss'])
		plt.plot(history.history['duration_out_loss'])
		plt.plot(history.history['syll_out_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Note', 'Duration', 'Syllable'], loc='upper left')
		plt.savefig("{}{}{}.png".format(params["plot_dir"],name,append_name))
		plt.show()

	"""
	Description: Generate a sequence from the trained neural network
	Params:
	- name (string): name of the run in order to load the weights
	- verbose (bool): boolean value to indicate whether to print the predicted elements or not
	Return:
	- a list with the sequence of notes, durations and syllables
	"""

	def generate_sequence(self, name, verbose = True):
		self.model.load_weights("{}{}".format(params["weights_dir"],name))
		
		# pick a random sequence from the input as a starting point for the prediction
		start = np.random.randint(0, len(self.in_notes_enc))
		pattern_note = []
		pattern_duration = []
		pattern_syll = []

		pattern_note.append(to_categorical(self.in_notes_enc[start][0], num_classes=self.notes_vocab_size))
		pattern_duration.append(to_categorical(self.in_duration_enc[start][0], num_classes=self.duration_vocab_size))
		pattern_syll.append(to_categorical(self.in_syll_enc[start][0], num_classes=self.syll_vocab_size))

		just_notes = []
		just_duration = []
		just_syll = []

		for note_index in range(params["sequence_length"] + 1):
			net_p_n = np.array(pattern_note).reshape(1,len(pattern_note),self.notes_vocab_size)
			net_p_d = np.array(pattern_duration).reshape(1,len(pattern_duration),self.duration_vocab_size)
			net_p_s = np.array(pattern_syll).reshape(1,len(pattern_syll),self.syll_vocab_size)

			pred = self.model.predict( [net_p_n, net_p_d, net_p_s], verbose=0)

			softmax_note = pred[0][0][note_index]
			i_pred_note = sample(softmax_note, 1)
			softmax_duration = pred[1][0][note_index]
			i_pred_duration = sample(softmax_duration, 1)
			softmax_syll = pred[2][0][note_index]
			i_pred_syll = sample(softmax_syll, 0)

			if verbose:
				print(pred_note, " | ", pred_duration, " | ", pred_syll)

			pattern_note.append(to_categorical(i_pred_note, num_classes=self.notes_vocab_size))
			pattern_duration.append(to_categorical(i_pred_duration, num_classes=self.duration_vocab_size))
			pattern_syll.append(to_categorical(i_pred_syll, num_classes=self.syll_vocab_size))

			just_notes.append(i_pred_note)
			just_duration.append(i_pred_duration)
			just_syll.append(i_pred_syll)

		return [just_notes, just_duration, just_syll]

	"""
	Description: Generate in bulk sequences from the trained neural network
	Params:
	- n (int): number of sequences to generate
	- name (string): name of the run in order to load the weights
	- out_dir (string): output directory for the generated MIDI files
	Return:
	- void
	"""

	def generate_bulk(self, n, name, out_dir):
		syll_idx = {val:index for index, val in enumerate(self.syll_vocab)}
		idx_syll = {index:val for index, val in enumerate(self.syll_vocab)}

		notes_idx = {val:index for index, val in enumerate(self.notes_vocab)}
		idx_notes = {index:val for index, val in enumerate(self.notes_vocab)}

		duration_idx = {val:index for index, val in enumerate(self.duration_vocab)}
		idx_duration = {index:val for index, val in enumerate(self.duration_vocab)}

		complete = ""

		for i in tqdm(range(n)):
			p = self.generate_sequence(name, verbose=False)

			p_note_dec = [idx_notes[n] for n in p[0]]
			p_duration_dec = [idx_duration[d] for d in p[1]]
			p_syll_dec = [idx_syll[s] for s in p[2]]

			create_midi(p_note_dec, p_duration_dec, "{}/{}.mid".format(out_dir, i))
			
			tab = score(p_note_dec, p_duration_dec, p_syll_dec)
			with open("{}/{}.txt".format(out_dir, i), "w") as score_file:
				score_file.write(tab)

			complete += "***** {} *****\n{}\n\n".format(i,tab)

			with open("{}/total.txt".format(out_dir), "w") as total_score_file:
				total_score_file.write(complete)