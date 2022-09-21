#coding:utf-8
import numpy as np
import tensorflow as tf

class Model(object):

	def get_config(self):
		return self.config

	def get_positive_instance(self, in_batch = True):
		if in_batch:
			return [self.positive_emb_h, self.positive_emb_t, self.positive_emb_r, self.positive_y, self.positive_dm_nbrs, self.positive_dm_nbrs_len, self.positive_rg_nbrs, self.positive_rg_nbrs_len]
		else:
			return [self.batch_emb_h[0: self.config.batch_size],
					self.batch_emb_t[0: self.config.batch_size],
					self.batch_emb_r[0: self.config.batch_size],					
					self.batch_y[0: self.config.batch_size], 
					self.batch_dm_nbrs[0:self.config.batch_size * self.config.hd_max],
					self.batch_dm_nbrs_len[0:self.config.batch_size],
					# self.batch_dm_nbrs_prob[0:self.config.batch_size * self.config.hd_max],
					self.batch_rg_nbrs[0:self.config.batch_size * self.config.tl_max],
					self.batch_rg_nbrs_len[0:self.config.batch_size]]
					# self.batch_rg_nbrs_prob[0:self.config.batch_size * self.config.tl_max]

	def get_negative_instance(self, in_batch = True):
		if in_batch:
			return [self.negative_emb_h, self.negative_emb_t, self.negative_emb_r, self.negative_y, 
					self.negative_dm_nbrs, self.negative_dm_nbrs_len, self.negative_rg_nbrs, self.negative_rg_nbrs_len]
		else:
			return [self.batch_emb_h[self.config.batch_size: self.config.batch_seq_size],
					self.batch_emb_t[self.config.batch_size: self.config.batch_seq_size],
					self.batch_emb_r[self.config.batch_size: self.config.batch_seq_size],
					self.batch_y[self.config.batch_size: self.config.batch_seq_size],

					self.batch_dm_nbrs[self.config.batch_size * self.config.hd_max:self.config.batch_seq_size * self.config.hd_max],
					self.batch_dm_nbrs_len[self.config.batch_size: self.config.batch_seq_size],
					# self.batch_dm_nbrs_prob[self.config.batch_size * self.config.hd_max:self.config.batch_seq_size * self.config.hd_max],
					self.batch_rg_nbrs[self.config.batch_size * self.config.tl_max:self.config.batch_seq_size * self.config.tl_max],
					self.batch_rg_nbrs_len[self.config.batch_size: self.config.batch_seq_size]]
					# self.batch_rg_nbrs_prob[self.config.batch_size * self.config.tl_max:self.config.batch_seq_size * self.config.tl_max]]

	def get_predict_instance(self):
		return [self.predict_emb_h, self.predict_emb_t, self.predict_emb_r, self.predict_dm_nbrs, self.predict_dm_nbrs_len, self.predict_rg_nbrs, self.predict_rg_nbrs_len]
		
	def input_def(self):
		config = self.config
		self.batch_emb_h = tf.placeholder(tf.int64, [config.batch_seq_size])
		self.batch_emb_t = tf.placeholder(tf.int64, [config.batch_seq_size])
		self.batch_emb_r = tf.placeholder(tf.int64, [config.batch_seq_size])
		self.batch_y = tf.placeholder(tf.float32, [config.batch_seq_size])

		self.batch_dm_nbrs = tf.placeholder(tf.int64, [config.batch_seq_size * self.config.hd_max])
		self.batch_dm_nbrs_len = tf.placeholder(tf.int64, [config.batch_seq_size])
		# self.batch_dm_nbrs_prob = tf.placeholder(tf.int64, [config.batch_seq_size * self.config.hd_max])
		self.batch_rg_nbrs = tf.placeholder(tf.int64, [config.batch_seq_size * self.config.tl_max])
		self.batch_rg_nbrs_len = tf.placeholder(tf.int64, [config.batch_seq_size])
		# self.batch_rg_nbrs_prob = tf.placeholder(tf.int64, [config.batch_seq_size * self.config.tl_max])

		self.positive_emb_h = tf.transpose(tf.reshape(self.batch_emb_h[0:config.batch_size], [1, -1]), perm = [1, 0]) #shape:[batch_size,1]
		self.positive_emb_t = tf.transpose(tf.reshape(self.batch_emb_t[0:config.batch_size], [1, -1]), perm = [1, 0])
		self.positive_emb_r = tf.transpose(tf.reshape(self.batch_emb_r[0:config.batch_size], [1, -1]), perm = [1, 0])
		self.positive_y = tf.transpose(tf.reshape(self.batch_y[0:config.batch_size], [1, -1]), perm = [1, 0])

		self.positive_dm_nbrs = tf.transpose(tf.reshape(self.batch_dm_nbrs[0:config.batch_size * config.hd_max], [1,-1]), perm=[1, 0]) # shape: [batch_size*hd_max, 1]
		self.positive_dm_nbrs_len = tf.transpose(tf.reshape(self.batch_dm_nbrs_len[0:config.batch_size], [1, -1]), perm=[1, 0])
		# self.positive_dm_nbrs_prob = tf.transpose(tf.reshape(self.batch_dm_nbrs_prob[0:config.batch_size * config.hd_max], [1,-1]), perm=[1, 0]) # shape: [batch_size*hd_max, 1]
		
		self.positive_rg_nbrs = tf.transpose(tf.reshape(self.batch_rg_nbrs[0:config.batch_size * config.tl_max], [1,-1]), perm=[1, 0])
		self.positive_rg_nbrs_len = tf.transpose(tf.reshape(self.batch_rg_nbrs_len[0:config.batch_size], [1,-1]), perm=[1, 0])
		# self.positive_rg_nbrs_prob = tf.transpose(tf.reshape(self.batch_rg_nbrs_prob[0:config.batch_size * config.tl_max], [1,-1]), perm=[1, 0])
		
		#neg
		self.negative_emb_h = tf.transpose(tf.reshape(self.batch_emb_h[config.batch_size:config.batch_seq_size], [config.negative_ent, -1]), perm = [1, 0]) 
		self.negative_emb_t = tf.transpose(tf.reshape(self.batch_emb_t[config.batch_size:config.batch_seq_size], [config.negative_ent, -1]), perm = [1, 0])
		self.negative_emb_r = tf.transpose(tf.reshape(self.batch_emb_r[config.batch_size:config.batch_seq_size], [config.negative_ent, -1]), perm = [1, 0])
		self.negative_y = tf.transpose(tf.reshape(self.batch_y[config.batch_size:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])

		self.negative_dm_nbrs = tf.transpose(tf.reshape(self.batch_dm_nbrs[config.batch_size * config.hd_max:config.batch_seq_size * config.hd_max], [config.negative_ent, -1]), perm=[1, 0])
		self.negative_dm_nbrs_len = tf.transpose(tf.reshape(self.batch_dm_nbrs_len[config.batch_size:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
		# self.negative_dm_nbrs_prob = tf.transpose(tf.reshape(self.batch_dm_nbrs_prob[config.batch_size * config.hd_max:config.batch_seq_size * config.hd_max], [1, -1]), perm=[1, 0])

		self.negative_rg_nbrs = tf.transpose(tf.reshape(self.batch_rg_nbrs[config.batch_size*config.tl_max : config.batch_seq_size*config.tl_max], [config.negative_ent,-1]), perm=[1, 0])
		self.negative_rg_nbrs_len = tf.transpose(tf.reshape(self.batch_rg_nbrs_len[config.batch_size: config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
		# self.negative_rg_nbrs_prob = tf.transpose(tf.reshape(self.batch_rg_nbrs_prob[config.batch_size*config.tl_max : config.batch_seq_size*config.tl_max], [1,-1]), perm=[1, 0])
		
		self.predict_emb_h = tf.placeholder(tf.int64, [None])
		self.predict_emb_t = tf.placeholder(tf.int64, [None])
		self.predict_emb_r = tf.placeholder(tf.int64, [None])
		self.predict_dm_nbrs = tf.placeholder(tf.int64, [None])
		self.predict_dm_nbrs_len = tf.placeholder(tf.int64, [None])
		# self.predict_dm_nbrs_prob = tf.placeholder(tf.int64, [None])
		self.predict_rg_nbrs = tf.placeholder(tf.int64, [None])
		self.predict_rg_nbrs_len = tf.placeholder(tf.int64, [None])
		# self.predict_rg_nbrs_prob = tf.placeholder(tf.int64, [None])

		self.parameter_lists = []

	def embedding_def(self):
		pass

	def loss_def(self):
		pass

	def predict_def(self):
		pass

	def __init__(self, config):
		print("in Model.py")
		self.config = config

		with tf.name_scope("input"):
			self.input_def() #placeholder for instances

		with tf.name_scope("embedding"): 
			self.embedding_def() #tf.get_variable

		with tf.name_scope("loss"):
			self.loss_def() #cal loss for training

		with tf.name_scope("predict"):
			self.predict_def() #prediction
