from __future__ import print_function

import numpy as np
from numpy.random import RandomState
from collections import Counter

np.random.seed(1)

def shuffle(data, random_seed=1):
	data_copy = np.copy(data).tolist()
	rand = RandomState(random_seed)
	rand.shuffle(data_copy)
	return data_copy

class BinaryData(object):

	def __init__(self, max_seq_len=50, size=100000, length_type="fixed"):
		length_options = ["fixed", "variable", "fair"]
		assert length_type in length_options, "length parameter should be one of {}".format(length_options)
		self.max_seq_len = max_seq_len
		self.size = size
		self.length_type = length_type
		self.data = []
		self.labels = []
		# Either 0 or 1 or padding character -1
		# eg. [1,0,1,-1,-1] if max_seq_len == 5
		self.dictionary = [0, 1, -1]
		if length_type == "fixed":
			self.data = (np.random.rand(self.size, self.max_seq_len) > 0.5).astype(np.int32)
			# XOR-operation returns False (0) if sum(sequence) == len(sequence) or sum(sequence) == 0
			# Else returns False (1)
			self.labels = (((np.sum(self.data, axis=1) == 0).astype(np.int32) + (np.sum(self.data, axis=1) == self.max_seq_len).astype(np.int32)) == 0).astype(np.int32)
		elif length_type == "variable":
			self.data = None
			self.labels = None
			counts = Counter(np.random.choice(np.arange(1, self.max_seq_len+1), self.size))
			for length, count in counts.items():
				subdata = (np.random.rand(count, length) > 0.5).astype(np.int32)
				sublabels = (((np.sum(subdata, axis=1) == 0).astype(np.int32) + (np.sum(subdata, axis=1) == length).astype(np.int32)) == 0).astype(np.int32)
				subdata = np.concatenate((subdata, (-1 * np.ones((count, self.max_seq_len - length)))), axis=1)
				if self.data is None:
					self.data = subdata
				else:
					self.data = np.concatenate((self.data, subdata))
				if self.labels is None:
					self.labels = sublabels
				else:
					self.labels = np.concatenate((self.labels, sublabels))
		elif length_type == "fair":
			self.data = None
			self.labels = None
			counts = Counter(np.random.choice(np.arange(1, self.max_seq_len+1), self.size))
			for length, count in counts.items():
				if length == 1:
					true_count = 0
					false_count = int(count)
				else:
					true_count = int(count / 2)
					false_count = count - true_count
				true_subdata = (np.random.rand(true_count, length) > 0.5).astype(np.int32)
				true_sublabels = (((np.sum(true_subdata, axis=1) == 0).astype(np.int32) + (np.sum(true_subdata, axis=1) == length).astype(np.int32)) == 0).astype(np.int32)
				while sum(true_sublabels) < len(true_sublabels):
					true_subdata[np.argmin(true_sublabels)] = (np.random.rand(length) > 0.5).astype(np.int32)
					true_sublabels = (((np.sum(true_subdata, axis=1) == 0).astype(np.int32) + (np.sum(true_subdata, axis=1) == length).astype(np.int32)) == 0).astype(np.int32)
				false_counts = Counter(np.random.choice([0, 1], false_count))
				false_subdata_0 = np.zeros((false_counts[0], length))
				false_subdata_1 = np.ones((false_counts[1], length))
				subdata = np.concatenate((true_subdata, false_subdata_0, false_subdata_1))
				sublabels = (((np.sum(subdata, axis=1) == 0).astype(np.int32) + (np.sum(subdata, axis=1) == length).astype(np.int32)) == 0).astype(np.int32)
				subdata = np.concatenate((subdata, (-1 * np.ones((count, self.max_seq_len - length)))), axis=1)
				if self.data is None:
					self.data = subdata
				else:
					self.data = np.concatenate((self.data, subdata))
				if self.labels is None:
					self.labels = sublabels
				else:
					self.labels = np.concatenate((self.labels, sublabels))

		self.data = shuffle(self.data)
		self.labels = shuffle(self.labels)
		self.idx = 0

	def pad(self, string):
		seq = list(string)
		for i in np.arange(self.max_seq_len - len(seq)):
			seq.append(-1)
		return seq

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

	def seq2vec(self, data):
		return np.eye(len(self.dictionary))[np.array(data, dtype=np.int32), :]
