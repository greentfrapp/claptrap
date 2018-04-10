from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from numpy.random import RandomState
import argparse
import time
import datetime
import os

from BinaryData import BinaryData

tf.set_random_seed(1)
np.random.seed(1)


def train_model(input_data, labels):
	latent_dim = 128
	batch_size = 10000
	n_epochs = 100
	id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
	model_folder = os.path.join("results", "{}_lstm_xor".format(id_no))
	tf.gfile.MakeDirs(model_folder)

	lstm_inputs = Input(shape=(None, input_data.shape[2]))
	lstm = LSTM(latent_dim)
	lstm_outputs = lstm(lstm_inputs)
	dense = Dense(1)
	outputs = dense(lstm_outputs)

	model = Model(lstm_inputs, outputs)
	print(model.summary())

	model.compile(optimizer='adam', loss='mean_squared_error')
	model.fit(input_data, labels,
		batch_size=batch_size,
		epochs=n_epochs,
		validation_split=0.2)

	model.save(os.path.join(model_folder, 'lstm_xor.h5'))

	return model

def train(hidden_num=128, batch_size=10000, max_seq_len=50, n_epochs=100, data_size=100000, length_type="fixed"):
	dataset = BinaryData(max_seq_len=max_seq_len, size=data_size, length_type=length_type)
	samples, labels = dataset.next(dataset.size)
	train_model(samples, labels)

def test(modelpath, max_seq_len=50, data_size=100000, length_type="fixed"):
	dataset = BinaryData(max_seq_len=max_seq_len, size=data_size, length_type=length_type)
	samples, labels = dataset.next(dataset.size)
	
	model = load_model(modelpath)
	
	predictions = (model.predict(samples) > 0.5).astype(np.float32).reshape(-1)

	print("Accuracy on '{}' dataset: {}".format(length_type, float(sum(labels == predictions)) / len(labels)))

def test_live(modelpath, max_seq_len=50):
	dataset = BinaryData(max_seq_len=max_seq_len, size=1)
	
	model = load_model(modelpath)
	print("\nEnter 'q' to quit...")
	while True:
		string = raw_input("\nInput         : ")
		if string == "q":
			break
		vector = np.expand_dims(dataset.seq2vec(dataset.pad(string)), axis=0)
		prediction = model.predict(vector)[0]
		if prediction > 0.5:
			xor_output = 1
		else:
			xor_output = 0
		print('Predicted XOR : {}'.format(xor_output))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Implementation of LSTM for XOR problem")
	# Arguments for training
	parser.add_argument("--train", action='store_true')
	parser.add_argument("-hn", "--hiddennum", action='store', type=int, default=128)
	parser.add_argument("-sl", "--seqlen", action='store', type=int, default=50)
	parser.add_argument("-ds", "--datasize", action='store', type=int, default=100000)
	parser.add_argument("-bs", "--batchsize", action='store', type=int, default=10000)
	parser.add_argument("-e", "--epoch", action='store', type=int, default=100)
	parser.add_argument("-type", "--lengthtype", action='store', choices=["fixed", "variable", "fair"], default="fixed")
	# Arguments for testing
	parser.add_argument("--test", action='store_true')
	parser.add_argument("--live", action='store_true')
	parser.add_argument("-m", "--modelpath", action='store')
	# parser.add_argument("-type", "--lengthtype", action='store', choices=["fixed", "variable", "fair"], default="fixed")
	args = parser.parse_args()
	if args.train:
		train(hidden_num=args.hiddennum, 
			batch_size=args.batchsize, 
			max_seq_len=args.seqlen, 
			n_epochs=args.epoch,
			data_size=args.datasize,
			length_type=args.lengthtype)
	elif args.test:
		modelpath = args.modelpath
		if not modelpath:
			print("\nNo path supplied, using latest model...")
			models = os.listdir("./results")
			models.sort()
			modelpath = os.path.join("results", models[-1], "lstm_xor.h5")
		print("\nUsing model from: {}\n".format(modelpath))
		if args.live:
			test_live(modelpath, max_seq_len=args.seqlen)
		else:
			test(modelpath, max_seq_len=args.seqlen, data_size=args.datasize, length_type=args.lengthtype)
	else:
		parser.print_help()
