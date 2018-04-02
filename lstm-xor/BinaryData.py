from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy.random import RandomState

tf.set_random_seed(1)
np.random.seed(1)

def shuffle(data, random_seed=1):
	data_copy = np.copy(data).tolist()
	rand = RandomState(random_seed)
	rand.shuffle(data_copy)
	return data_copy

class BinaryData(object):

	def __init__(self, max_seq_len=50, size=100000, length_type="fixed"):
		length_options = ["fixed", "variable"]
		assert length_type in length_options, "length parameter should be one of {}".format(length_options)
		self.max_seq_len = max_seq_len
		self.size = size
		self.length_type = length_type
		self.data = []
		self.labels = []
		# Either '1' or '0' or padding character '*'
		# eg. '101**' if max_seq_len == 5
		self.dictionary = ["1", "0", "*"]
		for i in np.arange(self.size):
			if length_type == "fixed":
				length = self.max_seq_len
			elif length_type == "variable":
				length = np.random.choice(np.arange(1, self.max_seq_len + 1))
			sequence = ""
			seq_sum = 0
			for j in np.arange(length):
				bit = np.random.choice([0, 1])
				sequence += str(bit)
				seq_sum += bit
			self.data.append(self.pad(sequence))
			# XOR-operation returns True (1) if sum(sequence) == len(sequence) or sum(sequence) == 0
			if seq_sum == len(sequence) or seq_sum == 0:
				self.labels.append([1])
			# Else returns False (0)
			else:
				self.labels.append([0])
		self.data = shuffle(self.data)
		self.labels = shuffle(self.labels)
		self.idx = 0

	def pad(self, sequence):
		for i in np.arange(self.max_seq_len - len(sequence)):
			sequence += "*"
		return sequence

	def next(self, batch_size):
		start = self.idx
		end = self.idx + batch_size
		if end >= self.size:
			end -= self.size
			batch_raw_data = self.data[start:]
			self.data = shuffle(self.data)
			batch_raw_data += self.data[:end]
			batch_labels = self.labels[start:]
			self.labels = shuffle(self.labels)
			batch_labels += self.labels[:end]
		else:
			batch_raw_data = self.data[start:end]
			batch_labels = self.labels[start:end]
		self.idx = end
		batch_data = self.seq2vec(batch_raw_data)
		return np.array(batch_data), np.array(batch_labels)

	def seq2vec(self, data, reverse=False):
		output = []
		for sample in data:
			sample = self.pad(sample)
			vector = []
			for c in list(sample):
				vector.append(np.eye(len(self.dictionary))[self.dictionary.index(c)])
			if reverse:
				vector = vector[::-1]
			output.append(vector)
		return output
