from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import argparse
import os

from LSTMClassifier import LSTMClassifier
from BinaryData import BinaryData

try: raw_input
except: raw_input = input

tf.set_random_seed(1)
np.random.seed(1)

def train(hidden_num=128, batch_size=10000, max_seq_len=50, n_epochs=100, data_size=100000, length_type="fixed"):
	dataset = BinaryData(max_seq_len=max_seq_len, size=data_size, length_type=length_type)
	classifier = LSTMClassifier(hidden_num=128, batch_size=10000, max_seq_len=dataset.max_seq_len, token_dim=len(dataset.dictionary))
	classifier.train(n_epochs=20, dataset=dataset)

def test(modelpath):
	classifier = LSTMClassifier(batch_size=1)
	classifier.load(modelpath)
	print("\nEnter 'q' to quit...")
	while True:
		seq = raw_input("\nInput         : ")
		if seq == "q":
			break
		vector = classifier.dataset.seq2vec([seq])
		prediction = classifier.test(vector)[0]
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
	parser.add_argument("-type", "--lengthtype", action='store', choices=["fixed", "variable"], default="fixed")
	# Arguments for testing
	parser.add_argument("--test", action='store_true')
	parser.add_argument("-m", "--modelpath", action='store')
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
			modelpath = os.path.join("results", os.listdir("./results")[-1])
		print("\nUsing model from: {}\n".format(modelpath))
		test(modelpath)
	else:
		parser.print_help()
