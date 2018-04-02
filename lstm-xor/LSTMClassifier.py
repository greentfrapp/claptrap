from __future__ import print_function

import tensorflow as tf
import numpy as np
import datetime
import time
import os
import json

from BinaryData import BinaryData

class LSTMClassifier(object):

	def __init__(self, hidden_num=128, batch_size=10000, max_seq_len=50, token_dim=3):
		self.hidden_num = hidden_num
		self.batch_size = batch_size
		self.max_seq_len = max_seq_len
		self.token_dim = token_dim

		# Placeholders
		self.inputs = tf.placeholder(shape=[None, self.max_seq_len, self.token_dim], dtype=tf.float32)
		self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

		# LSTM Cell
		self._enc_cell = tf.contrib.rnn.LSTMCell(hidden_num)
		self.transformed_inputs = [tf.squeeze(t, [1]) for t in tf.split(self.inputs, self.max_seq_len, 1)]
		self.batch_enc_output, self.batch_enc_state = tf.contrib.rnn.static_rnn(self._enc_cell, self.transformed_inputs, dtype=tf.float32)
		self.batch_enc_output = tf.stack(self.batch_enc_output)
		self.batch_enc_output = tf.transpose(self.batch_enc_output, [1, 0, 2])
		index = tf.range(0, batch_size) * max_seq_len + max_seq_len - 1
		self.batch_enc_output = tf.gather(tf.reshape(self.batch_enc_output, [-1, self.hidden_num]), index)

		# Fully-connected Layer
		weights = tf.Variable(tf.truncated_normal([self.hidden_num, 1], dtype=tf.float32))
		bias = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float32))

		self.output = tf.matmul(self.batch_enc_output, weights) + bias

		# Loss Function and Optimizer
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.labels))
		self.optimize = tf.train.AdamOptimizer().minimize(self.loss)

		# Predictions
		self.predictions = tf.nn.sigmoid(self.output)
		
		# Tensorboard Info
		tf.summary.scalar(name="Training Loss", tensor=self.loss)
		self.summary = tf.summary.merge_all()

		# Initialize Session, variables and Saver
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def create_checkpoint_folders(self):
		self.folder = "{}_lstm_xor".format(self.id_no)
		subfolders = ["tensorboard/", "saved_models/", "log/"]
		paths = ()
		for subfolder in subfolders:
			path = os.path.join(self.results_path, self.folder, subfolder)
			tf.gfile.MakeDirs(path)
			paths += (path,)
		return paths

	def load(self, modelpath, batch_size=None):
		with open(os.path.join(modelpath, "details.json"), 'r') as file:
			self.details = json.load(file)
		self.hidden_num = self.details["hidden_num"]
		if batch_size is None:
			self.batch_size = self.details["batch_size"]
		else:
			self.batch_size = batch_size
		self.max_seq_len = self.details["max_seq_len"]
		self.n_epochs = self.details["n_epochs"]
		self.dataset = BinaryData(max_seq_len=self.max_seq_len, size=self.details["data_size"], length_type=self.details["length_type"])
		self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(os.path.join(modelpath, "saved_models")))
		return None

	def print_log(self, epoch, n_epochs, loss):
		entry = "Epoch {}/{} - loss {}".format(epoch, n_epochs, loss)
		print(entry)
		with open(self.log_path + '/log.txt', 'a') as log:
			log.write(entry)

	def save_details(self):
		self.details = dict()
		self.details["hidden_num"] = self.hidden_num
		self.details["batch_size"] = self.batch_size
		self.details["max_seq_len"] = self.max_seq_len
		self.details["n_epochs"] = self.n_epochs
		self.details["data_size"] = self.dataset.size
		self.details["length_type"] = self.dataset.length_type
		with open(os.path.join(self.results_path, self.folder, "details.json"), 'w') as file:
			json.dump(self.details, file)

	def train(self, n_epochs, dataset):
		self.id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
		self.n_epochs = n_epochs
		self.dataset = dataset

		# Create results_folder
		self.results_path = 'results'
		tf.gfile.MakeDirs(self.results_path)

		self.step = 0
		self.tensorboard_path, self.modelpath, self.log_path = self.create_checkpoint_folders()
		print(self.create_checkpoint_folders())
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)
		self.save_details()

		print("\nTraining Parameters:")
		for item, value in self.details.items():
			print("{} : {}".format(item, value))
		print("")
		
		for i in np.arange(n_epochs):
			n_iters = self.dataset.size / self.batch_size
			for j in np.arange(n_iters):
				input_data, input_labels = self.dataset.next(self.batch_size)
				loss, summary, _ = self.sess.run([self.loss, self.summary, self.optimize], feed_dict={self.inputs: input_data, self.labels: input_labels})
				self.step += 1
				self.writer.add_summary(summary, global_step=self.step)
			self.print_log(i + 1, n_epochs, loss)
			self.saver.save(self.sess, save_path=self.modelpath, global_step=self.step)
		return None

	def test(self, sample):
		return self.sess.run(self.predictions, feed_dict={self.inputs: sample})
