#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import random
from sklearn.metrics import roc_curve, auc 
from sklearn import preprocessing

class Config(object):
	'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base.so'))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		# training
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] 

		# link prediction
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] 
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p]
		self.lib.testTail.argtypes = [ctypes.c_void_p]

		# triple classification
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] 
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

		# noise detection
		self.lib.getNoiPosBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] 
		self.lib.getNoiNegBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

		self.test_flag = False
		self.in_path = None
		self.neg_path = None
		self.out_path = None
		self.bern = 0
		self.type_dim = 20
		self.sem_dim = 50
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.eval_batch = 500
		self.valid_batch = 500
		self.test_batch = 500
		self.workThreads = 1
		self.negative_ent = 1
		self.alpha = 0.001
		self.log_on = 1
		self.exportName = "./ckpt/test-model.ckpt"
		# self.exportName = None
		self.s_rate = 0.1
		self.k_rate = 0.1
		self.importName = None
		self.export_steps = 0
		self.data_dir = "./"
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = True
		self.test_triple_classification = False
		self.test_inconsistency_detection = False
		self.test_noise_detection = False
		self.early_stopping = None

	def init_link_prediction(self):
		self.lib.importTestFiles()
		self.lp_test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.lp_test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.lp_test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.lp_test_dm_nbrs = np.zeros(self.lib.getEntityTotal()*self.hd_max, dtype = np.int64)
		self.lp_test_dm_nbrs_len = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.lp_test_rg_nbrs = np.zeros(self.lib.getEntityTotal()*self.tl_max, dtype = np.int64)
		self.lp_test_rg_nbrs_len = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)

		self.lp_test_h_addr = self.lp_test_h.__array_interface__['data'][0]
		self.lp_test_t_addr = self.lp_test_t.__array_interface__['data'][0]
		self.lp_test_r_addr = self.lp_test_r.__array_interface__['data'][0]
		self.lp_test_dm_nbrs_addr = self.lp_test_dm_nbrs.__array_interface__['data'][0]
		self.lp_test_dm_nbrs_len_addr = self.lp_test_dm_nbrs_len.__array_interface__['data'][0]
		self.lp_test_rg_nbrs_addr = self.lp_test_rg_nbrs.__array_interface__['data'][0]
		self.lp_test_rg_nbrs_len_addr = self.lp_test_rg_nbrs_len.__array_interface__['data'][0]
		
	# triple classification
	def init_triple_classification(self):
		r'''
		import essential files and set essential interfaces for triple classification
		'''
		self.lib.importTestFiles()
		self.validTotal = self.lib.getValidTotal()
		self.testTotal = self.lib.getTestTotal()
		self.hd_max = self.lib.getHd_max()
		self.tl_max = self.lib.getTl_max()
		
		self.test_pos_emb_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_emb_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_emb_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_dm_nbrs = np.zeros(self.lib.getTestTotal() * self.hd_max, dtype=np.int64)
		self.test_pos_dm_nbrs_len = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_rg_nbrs = np.zeros(self.lib.getTestTotal() * self.tl_max, dtype=np.int64)
		self.test_pos_rg_nbrs_len = np.zeros(self.lib.getTestTotal(), dtype = np.int64)

		self.test_neg_emb_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_emb_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_emb_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_dm_nbrs = np.zeros(self.lib.getTestTotal() * self.hd_max, dtype=np.int64)
		self.test_neg_dm_nbrs_len = np.zeros(self.lib.getTestTotal(), dtype=np.int64)
		self.test_neg_rg_nbrs = np.zeros(self.lib.getTestTotal() * self.tl_max, dtype=np.int64)
		self.test_neg_rg_nbrs_len = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		
		self.test_pos_emb_h_addr = self.test_pos_emb_h.__array_interface__['data'][0]
		self.test_pos_emb_t_addr = self.test_pos_emb_t.__array_interface__['data'][0]
		self.test_pos_emb_r_addr = self.test_pos_emb_r.__array_interface__['data'][0]
		self.test_pos_dm_nbrs_addr = self.test_pos_dm_nbrs.__array_interface__['data'][0]
		self.test_pos_dm_nbrs_len_addr = self.test_pos_dm_nbrs_len.__array_interface__['data'][0]
		self.test_pos_rg_nbrs_addr = self.test_pos_rg_nbrs.__array_interface__['data'][0]
		self.test_pos_rg_nbrs_len_addr = self.test_pos_rg_nbrs_len.__array_interface__['data'][0]

		self.test_neg_emb_h_addr = self.test_neg_emb_h.__array_interface__['data'][0]
		self.test_neg_emb_t_addr = self.test_neg_emb_t.__array_interface__['data'][0]
		self.test_neg_emb_r_addr = self.test_neg_emb_r.__array_interface__['data'][0]
		self.test_neg_dm_nbrs_addr = self.test_neg_dm_nbrs.__array_interface__['data'][0]
		self.test_neg_dm_nbrs_len_addr = self.test_neg_dm_nbrs_len.__array_interface__['data'][0]
		self.test_neg_rg_nbrs_addr = self.test_neg_rg_nbrs.__array_interface__['data'][0]
		self.test_neg_rg_nbrs_len_addr = self.test_neg_rg_nbrs_len.__array_interface__['data'][0]

		self.valid_pos_emb_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_emb_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_emb_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_dm_nbrs = np.zeros(self.lib.getValidTotal() * self.hd_max, dtype = np.int64)
		self.valid_pos_dm_nbrs_len = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_rg_nbrs = np.zeros(self.lib.getValidTotal() * self.tl_max, dtype=np.int64)
		self.valid_pos_rg_nbrs_len = np.zeros(self.lib.getValidTotal(), dtype=np.int64)

		self.valid_pos_emb_h_addr = self.valid_pos_emb_h.__array_interface__['data'][0]
		self.valid_pos_emb_t_addr = self.valid_pos_emb_t.__array_interface__['data'][0]
		self.valid_pos_emb_r_addr = self.valid_pos_emb_r.__array_interface__['data'][0]
		self.valid_pos_dm_nbrs_addr = self.valid_pos_dm_nbrs.__array_interface__['data'][0]
		self.valid_pos_dm_nbrs_len_addr = self.valid_pos_dm_nbrs_len.__array_interface__['data'][0]
		self.valid_pos_rg_nbrs_addr = self.valid_pos_rg_nbrs.__array_interface__['data'][0]
		self.valid_pos_rg_nbrs_len_addr = self.valid_pos_rg_nbrs_len.__array_interface__['data'][0]

		self.valid_neg_emb_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_emb_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_emb_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_dm_nbrs = np.zeros(self.lib.getValidTotal() * self.hd_max, dtype=np.int64)
		self.valid_neg_dm_nbrs_len = np.zeros(self.lib.getValidTotal(), dtype=np.int64)
		self.valid_neg_rg_nbrs = np.zeros(self.lib.getValidTotal() * self.tl_max, dtype=np.int64)
		self.valid_neg_rg_nbrs_len = np.zeros(self.lib.getValidTotal(), dtype=np.int64)

		self.valid_neg_emb_h_addr = self.valid_neg_emb_h.__array_interface__['data'][0]
		self.valid_neg_emb_t_addr = self.valid_neg_emb_t.__array_interface__['data'][0]
		self.valid_neg_emb_r_addr = self.valid_neg_emb_r.__array_interface__['data'][0]
		self.valid_neg_dm_nbrs_addr = self.valid_neg_dm_nbrs.__array_interface__['data'][0]
		self.valid_neg_dm_nbrs_len_addr = self.valid_neg_dm_nbrs_len.__array_interface__['data'][0]
		self.valid_neg_rg_nbrs_addr = self.valid_neg_rg_nbrs.__array_interface__['data'][0]
		self.valid_neg_rg_nbrs_len_addr = self.valid_neg_rg_nbrs_len.__array_interface__['data'][0]

		self.totThresh = np.zeros(self.lib.getRelationTotal(), dtype=np.float32)
		self.totThresh_addr = self.totThresh.__array_interface__['data'][0]

	# noise detection
	def init_noise_detection(self):
		self.lib.importTestFiles()
		self.noi_emb_h = np.zeros(self.lib.getTrainPosTotal(), dtype=np.int64)
		self.noi_emb_t = np.zeros(self.lib.getTrainPosTotal(), dtype=np.int64)
		self.noi_emb_r = np.zeros(self.lib.getTrainPosTotal(), dtype=np.int64)
		self.noi_dm_nbrs = np.zeros(self.lib.getTrainPosTotal() * self.hd_max, dtype=np.int64)
		self.noi_dm_nbrs_len = np.zeros(self.lib.getTrainPosTotal(), dtype=np.int64)
		self.noi_rg_nbrs = np.zeros(self.lib.getTrainPosTotal() * self.tl_max, dtype=np.int64)
		self.noi_rg_nbrs_len = np.zeros(self.lib.getTrainPosTotal(), dtype=np.int64)

		self.noi_emb_h_addr = self.noi_emb_h.__array_interface__['data'][0]
		self.noi_emb_t_addr = self.noi_emb_t.__array_interface__['data'][0]
		self.noi_emb_r_addr = self.noi_emb_r.__array_interface__['data'][0]
		self.noi_dm_nbrs_addr = self.noi_dm_nbrs.__array_interface__['data'][0]
		self.noi_dm_nbrs_len_addr = self.noi_dm_nbrs_len.__array_interface__['data'][0]
		self.noi_rg_nbrs_addr = self.noi_rg_nbrs.__array_interface__['data'][0]
		self.noi_rg_nbrs_len_addr = self.noi_rg_nbrs_len.__array_interface__['data'][0]
		
		self.noi_neg_emb_h = np.zeros(self.lib.getTrainNoiTotal(), dtype=np.int64)
		self.noi_neg_emb_t = np.zeros(self.lib.getTrainNoiTotal(), dtype=np.int64)
		self.noi_neg_emb_r = np.zeros(self.lib.getTrainNoiTotal(), dtype=np.int64)
		self.noi_neg_dm_nbrs = np.zeros(self.lib.getTrainNoiTotal() * self.hd_max, dtype=np.int64)
		self.noi_neg_dm_nbrs_len = np.zeros(self.lib.getTrainNoiTotal(), dtype=np.int64)
		self.noi_neg_rg_nbrs = np.zeros(self.lib.getTrainNoiTotal() * self.tl_max, dtype=np.int64)
		self.noi_neg_rg_nbrs_len = np.zeros(self.lib.getTrainNoiTotal(), dtype=np.int64)
		
		self.noi_neg_emb_h_addr = self.noi_neg_emb_h.__array_interface__['data'][0]
		self.noi_neg_emb_t_addr = self.noi_neg_emb_t.__array_interface__['data'][0]
		self.noi_neg_emb_r_addr = self.noi_neg_emb_r.__array_interface__['data'][0]
		self.noi_neg_dm_nbrs_addr = self.noi_neg_dm_nbrs.__array_interface__['data'][0]
		self.noi_neg_dm_nbrs_len_addr = self.noi_neg_dm_nbrs_len.__array_interface__['data'][0]
		self.noi_neg_rg_nbrs_addr = self.noi_neg_rg_nbrs.__array_interface__['data'][0]
		self.noi_neg_rg_nbrs_len_addr = self.noi_neg_rg_nbrs_len.__array_interface__['data'][0]

	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib.setNegPath(ctypes.create_string_buffer(self.neg_path.encode(), len(self.neg_path) * 2))
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset()
			self.lib.importTrainFiles()
			self.relTotal = self.lib.getRelationTotal()
			self.hd_max = self.lib.getHd_max()
			self.tl_max = self.lib.getTl_max()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			self.batch_size = int(self.lib.getTrainTotal() / self.nbatches)

			self.batch_seq_size = self.batch_size * (1 + self.negative_ent) 
			self.batch_emb_h = np.zeros(self.batch_seq_size, dtype = np.int64)
			self.batch_emb_t = np.zeros(self.batch_seq_size, dtype = np.int64)
			self.batch_emb_r = np.zeros(self.batch_seq_size, dtype = np.int64)
			self.batch_y = np.zeros(self.batch_seq_size, dtype = np.float32)
			
			self.batch_dm_nbrs = np.zeros(self.batch_seq_size * self.hd_max, dtype=np.int64)
			self.batch_dm_nbrs_len = np.zeros(self.batch_seq_size, dtype=np.int64)
			# self.batch_dm_nbrs_prob = np.zeros(self.batch_seq_size * self.hd_max, dtype=np.int64)
			self.batch_rg_nbrs = np.zeros(self.batch_seq_size * self.tl_max, dtype=np.int64)
			self.batch_rg_nbrs_len = np.zeros(self.batch_seq_size, dtype=np.int64)
			# self.batch_rg_nbrs_prob = np.zeros(self.batch_seq_size * self.tl_max, dtype=np.int64)
			
			self.batch_emb_h_addr = self.batch_emb_h.__array_interface__['data'][0]
			self.batch_emb_t_addr = self.batch_emb_t.__array_interface__['data'][0]
			self.batch_emb_r_addr = self.batch_emb_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]

			self.batch_dm_nbrs_addr = self.batch_dm_nbrs.__array_interface__['data'][0]
			self.batch_dm_nbrs_len_addr = self.batch_dm_nbrs_len.__array_interface__['data'][0]
			# self.batch_dm_nbrs_prob_addr = self.batch_dm_nbrs_prob.__array_interface__['data'][0]
			self.batch_rg_nbrs_addr = self.batch_rg_nbrs.__array_interface__['data'][0]
			self.batch_rg_nbrs_len_addr = self.batch_rg_nbrs_len.__array_interface__['data'][0]
			# self.batch_rg_nbrs_prob_addr = self.batch_rg_nbrs_prob.__array_interface__['data'][0]

		if self.test_link_prediction:
			self.init_link_prediction()

		if self.test_triple_classification:
			self.init_triple_classification()

		if self.test_noise_detection:
			self.init_noise_detection()

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_test_inconsistency_detection(self, flag):
		self.test_inconsistency_detection = flag

	def set_test_noise_detection(self, flag):
		self.test_noise_detection = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_bern(self, bern):
		self.bern = bern

	def set_tp_dimension(self, dim):
		self.type_dim = dim

	def set_sm_dimension(self, dim):
		self.sem_dim = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_data_dir(self, data_dir):
		self.data_dir = data_dir

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_in_path(self, path):
		self.in_path = path

	def set_neg_path(self, path):
		self.neg_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps = 0):
		self.exportName = path
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	def set_irr_num(self, irr_num):
		self.irrhtNum = irr_num

	def set_scorerate(self, s_rate):
		self.s_rate = s_rate

	def set_kgerate(self, k_rate):
		self.k_rate = k_rate

	def set_lossrate(self, l_rate):
		self.l_rate = l_rate

	# call C function for sampling
	def sampling(self):
		self.lib.sampling(self.batch_emb_h_addr, self.batch_emb_t_addr, self.batch_emb_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, 
			self.batch_dm_nbrs_addr, self.batch_dm_nbrs_len_addr, self.batch_rg_nbrs_addr, self.batch_rg_nbrs_len_addr)
	# save model
	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)
	# restore model
	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.exportName)

	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))#save parameters
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			#self.sess = tf.Session()
			config_setting = tf.ConfigProto()
			config_setting.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config_setting) 
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					if self.optimizer != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
					self.train_op = self.optimizer.apply_gradients(grads_and_vars)
				self.saver = tf.train.Saver()
				self.sess.run(tf.global_variables_initializer())

	def train_step(self, batch_emb_h, batch_emb_t, batch_emb_r, batch_y, batch_dm_nbrs, batch_dm_nbrs_len, batch_rg_nbrs, batch_rg_nbrs_len):
		feed_dict = {
			self.trainModel.batch_emb_h: batch_emb_h,
			self.trainModel.batch_emb_t: batch_emb_t,
			self.trainModel.batch_emb_r: batch_emb_r,
			self.trainModel.batch_y: batch_y,
			self.trainModel.batch_dm_nbrs: batch_dm_nbrs,
			self.trainModel.batch_dm_nbrs_len: batch_dm_nbrs_len,
			# self.trainModel.batch_dm_nbrs_prob: batch_dm_nbrs_prob,
			self.trainModel.batch_rg_nbrs: batch_rg_nbrs,
			self.trainModel.batch_rg_nbrs_len: batch_rg_nbrs_len
			# self.trainModel.batch_rg_nbrs_prob: batch_rg_nbrs_prob
			}
		_, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
		return loss

	def test_step(self, test_emb_h, test_emb_t, test_emb_r, test_dm_nbrs, test_dm_nbrs_len, test_rg_nbrs, test_rg_nbrs_len):
		feed_dict = {
			self.trainModel.predict_emb_h: test_emb_h,
			self.trainModel.predict_emb_t: test_emb_t,
			self.trainModel.predict_emb_r: test_emb_r,
			self.trainModel.predict_dm_nbrs: test_dm_nbrs,
			self.trainModel.predict_dm_nbrs_len: test_dm_nbrs_len,
			self.trainModel.predict_rg_nbrs: test_rg_nbrs,
			self.trainModel.predict_rg_nbrs_len: test_rg_nbrs_len
		}
		pred_tot, pred_emb, pred_dm, pred_rg, pred_dis, pred_irr, pred_asy = self.sess.run([self.trainModel.pred_emb_and_axiom, self.trainModel.pred_emb, self.trainModel.out_dm, self.trainModel.out_rg, self.trainModel.out_dis, self.trainModel.out_irr, self.trainModel.out_asy], feed_dict)
		return pred_tot, pred_emb, pred_dm, pred_rg, pred_dis, pred_irr, pred_asy

	def run(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.early_stopping is not None:
					patience, min_delta = self.early_stopping
					best_loss = np.finfo('float32').max
					wait_steps = 0
				for times in range(self.train_times):
					loss = 0.0
					t_init = time.time()
					for batch in range(self.nbatches):
						self.sampling()
						loss += self.train_step(self.batch_emb_h, self.batch_emb_t, self.batch_emb_r, self.batch_y, \
						self.batch_dm_nbrs, self.batch_dm_nbrs_len, self.batch_rg_nbrs, self.batch_rg_nbrs_len)
					t_end = time.time()
					if self.log_on:
						print('Epoch: {}, loss: {}, time: {}'.format(times, loss, (t_end - t_init)))
					if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
						self.save_tensorflow()
					if self.early_stopping is not None:
						if loss + min_delta < best_loss:
							best_loss = loss
							wait_steps = 0
						elif wait_steps < patience:
							wait_steps += 1
						else:
							print('Early stopping. Losses have not been improved enough in {} times'.format(patience))
							break
				if self.exportName != None:
					self.save_tensorflow() 
				if self.out_path != None:
					self.save_parameters(self.out_path)

	def test(self):
		with self.graph.as_default():
			with self.sess.as_default():
				# if self.importName != None:
				# 	self.restore_tensorflow()
				self.restore_tensorflow()
				if self.test_link_prediction:
					total = self.lib.getTestTotal()
					for times in range(total):
						self.lib.getHeadBatch(self.lp_test_h_addr, self.lp_test_t_addr, self.lp_test_r_addr, self.lp_test_dm_nbrs_addr, self.lp_test_dm_nbrs_len_addr, self.lp_test_rg_nbrs_addr, self.lp_test_rg_nbrs_len_addr)
						res_head, _, _, _, _, _, _ = self.test_step(self.lp_test_h, self.lp_test_t, self.lp_test_r, self.lp_test_dm_nbrs, self.lp_test_dm_nbrs_len, self.lp_test_rg_nbrs, self.lp_test_rg_nbrs_len)
						self.lib.testHead(res_head.__array_interface__['data'][0])

						self.lib.getTailBatch(self.lp_test_h_addr, self.lp_test_t_addr, self.lp_test_r_addr, self.lp_test_dm_nbrs_addr, self.lp_test_dm_nbrs_len_addr, self.lp_test_rg_nbrs_addr, self.lp_test_rg_nbrs_len_addr)
						res_tail, _, _, _, _, _, _ = self.test_step(self.lp_test_h, self.lp_test_t, self.lp_test_r, self.lp_test_dm_nbrs, self.lp_test_dm_nbrs_len, self.lp_test_rg_nbrs, self.lp_test_rg_nbrs_len)
						self.lib.testTail(res_tail.__array_interface__['data'][0])
						if self.log_on:
							print(times)
					self.lib.test_link_prediction()

				if self.test_triple_classification:
					self.lib.getValidBatch(self.valid_pos_emb_h_addr, self.valid_pos_emb_t_addr, self.valid_pos_emb_r_addr, self.valid_pos_dm_nbrs_addr, self.valid_pos_dm_nbrs_len_addr, self.valid_pos_rg_nbrs_addr, self.valid_pos_rg_nbrs_len_addr, 
										self.valid_neg_emb_h_addr, self.valid_neg_emb_t_addr, self.valid_neg_emb_r_addr, self.valid_neg_dm_nbrs_addr, self.valid_neg_dm_nbrs_len_addr, self.valid_neg_rg_nbrs_addr, self.valid_neg_rg_nbrs_len_addr)		
					res_pos_tot, _, _, _, _, _, _ = self.test_step(self.valid_pos_emb_h, self.valid_pos_emb_t, self.valid_pos_emb_r, self.valid_pos_dm_nbrs, self.valid_pos_dm_nbrs_len, self.valid_pos_rg_nbrs, self.valid_pos_rg_nbrs_len)

					res_neg_tot, _, _, _, _, _, _ = self.test_step(self.valid_neg_emb_h, self.valid_neg_emb_t, self.valid_neg_emb_r, self.valid_neg_dm_nbrs, self.valid_neg_dm_nbrs_len, self.valid_neg_rg_nbrs, self.valid_neg_rg_nbrs_len)
						
					self.lib.getBestThreshold(self.totThresh_addr, res_pos_tot.__array_interface__['data'][0], res_neg_tot.__array_interface__['data'][0])
					
					self.lib.getTestBatch(self.test_pos_emb_h_addr, self.test_pos_emb_t_addr, self.test_pos_emb_r_addr, self.test_pos_dm_nbrs_addr, self.test_pos_dm_nbrs_len_addr, self.test_pos_rg_nbrs_addr, self.test_pos_rg_nbrs_len_addr, 
					   self.test_neg_emb_h_addr, self.test_neg_emb_t_addr, self.test_neg_emb_r_addr, self.test_neg_dm_nbrs_addr, self.test_neg_dm_nbrs_len_addr, self.test_neg_rg_nbrs_addr, self.test_neg_rg_nbrs_len_addr)

					res_pos_tot_tt, res_pos_emb_tt, res_pos_dm_tt, res_pos_rg_tt, res_pos_dis_tt, res_pos_irr_tt, res_pos_asym_tt = self.test_step(self.test_pos_emb_h, self.test_pos_emb_t, self.test_pos_emb_r, self.test_pos_dm_nbrs, self.test_pos_dm_nbrs_len, self.test_pos_rg_nbrs, self.test_pos_rg_nbrs_len)

					res_neg_tot_tt, res_neg_emb_tt, res_neg_dm_tt, res_neg_rg_tt, res_neg_dis_tt, res_neg_irr_tt, res_neg_asym_tt  = self.test_step(self.test_neg_emb_h, self.test_neg_emb_t, self.test_neg_emb_r, self.test_neg_dm_nbrs, self.test_neg_dm_nbrs_len, self.test_neg_rg_nbrs, self.test_neg_rg_nbrs_len)

					self.lib.test_triple_classification(self.totThresh_addr, res_pos_tot_tt.__array_interface__['data'][0], res_neg_tot_tt.__array_interface__['data'][0])

				if self.test_noise_detection:
					self.lib.getNoiPosBatch(self.noi_emb_h_addr, self.noi_emb_t_addr, self.noi_emb_r_addr, self.noi_dm_nbrs_addr, self.noi_dm_nbrs_len_addr, self.noi_rg_nbrs_addr, self.noi_rg_nbrs_len_addr)
					self.lib.getNoiNegBatch(self.noi_neg_emb_h_addr, self.noi_neg_emb_t_addr, self.noi_neg_emb_r_addr, self.noi_neg_dm_nbrs_addr, self.noi_neg_dm_nbrs_len_addr, self.noi_neg_rg_nbrs_addr, self.noi_neg_rg_nbrs_len_addr)
	
					noi_pos_tot, _, _, _, _, _, _ = self.test_step(self.noi_emb_h, self.noi_emb_t, self.noi_emb_r, self.noi_dm_nbrs, self.noi_dm_nbrs_len, self.noi_rg_nbrs, self.noi_rg_nbrs_len)
					
					noi_neg_tot, _, _, _, _, _, _ = self.test_step(self.noi_neg_emb_h, self.noi_neg_emb_t, self.noi_neg_emb_r, self.noi_neg_dm_nbrs, self.noi_neg_dm_nbrs_len, self.noi_neg_rg_nbrs, self.noi_neg_rg_nbrs_len)
					
					print("positive triples=%ld" % self.lib.getTrainPosTotal())
					print("noise triples=%ld" % self.lib.getTrainNoiTotal()) 
					sess = tf.Session()
					label_pos = tf.ones([self.lib.getTrainPosTotal(), 1], dtype=tf.int64)
					label_neg = tf.zeros([self.lib.getTrainNoiTotal(), 1], dtype=tf.int64)
					label_data = sess.run(tf.cast(tf.concat([label_pos, label_neg], 0), dtype=tf.float32))
					noi_data = sess.run(tf.concat([noi_pos_tot, noi_neg_tot], 0)).reshape(-1,1)
					noi_data_normalize = 1-preprocessing.MinMaxScaler().fit_transform(noi_data)
					fpr, tpr, thresholds = roc_curve(label_data, noi_data_normalize, pos_label=1)
					noi_value = auc(fpr, tpr)
					print("noi_auc_value_all = %lf" % noi_value)

