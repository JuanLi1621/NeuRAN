#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model
import random

class NbrAttention_transe(Model):

	def _calc(self, h, t, r):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		return abs(h + r - t)
		
	def embedding_def(self):
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations
		self.ent_c_embeddings = tf.get_variable(name = "ent_c_embeddings", shape = [config.entTotal, config.type_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.ent_sm_embeddings = tf.get_variable(name = "ent_sm_embeddings", shape = [config.entTotal, config.sem_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_ss_embeddings = tf.get_variable(name="rel_ss_embeddings", shape=[(config.relTotal+1), config.type_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		self.rel_oo_embeddings = tf.get_variable(name="rel_oo_embeddings", shape=[(config.relTotal+1), config.type_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		self.rel_sm_embeddings = tf.get_variable(name="rel_sm_embeddings", shape=[(config.relTotal+1), config.sem_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		self.irr_w = tf.get_variable(name='irr_w', shape=[config.sem_dim+config.type_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		self.asym_w = tf.get_variable(name='asym_w', shape=[config.sem_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		self.parameter_lists = {"ent_c_embeddings":self.ent_c_embeddings, "ent_sm_embeddings":self.ent_sm_embeddings, 
								"rel_sm_embeddings":self.rel_sm_embeddings, "rel_ss_embeddings":self.rel_ss_embeddings, "rel_oo_embeddings":self.rel_oo_embeddings, 
								"irr_w":self.irr_w, "asym_w":self.asym_w}

	#input_embeddings: embedding of the current relation (bz, 1, type_dim)
	#nbr_embeddings: embedding of all relations (rel_num+1, type_dim)
	#nbrs: all nbrs, (bz*max_len, 1)
	def _aggregation(self, nbr_embeddings, input_embeddings, nbrs, hd_max):
		config = self.get_config()
		mask_emb = tf.concat([tf.ones([config.relTotal, 1]), tf.zeros([1,1])], 0) #(rel_num+1,0),[[1],[1],...,[1],[0]]
		mask_weight = tf.concat([tf.zeros([config.relTotal, 1]), tf.ones([1,1])*1e19], 0) #(rel_num+1,0), [[0],[0],...,[0],[1e19]]
		mask = tf.nn.embedding_lookup(mask_emb, tf.reshape(nbrs, [-1, hd_max])) #(bz,max_len,1)
		pos_dm_nbrs_vec = tf.nn.embedding_lookup(nbr_embeddings, tf.reshape(nbrs, [-1, hd_max])) #(bz,max_len, type_dim)
		pos_dm_nbrs_vec = pos_dm_nbrs_vec * mask #(bz, max_len, type_dim)
		pos_cal_att = tf.cast(tf.reshape(tf.matmul(pos_dm_nbrs_vec, tf.transpose(input_embeddings,perm=[0,2,1])), [-1, hd_max]), dtype=tf.float32) #(bz,max_len,1)->(bz,max_len)
		mask_att = tf.reshape(tf.nn.embedding_lookup(mask_weight, tf.reshape(nbrs, [-1, hd_max])), [-1, hd_max]) #(bz,max_len,1)->(bz,max_len)
		pos_cal_att -= mask_att
		pos_cal_att_final = tf.nn.softmax(pos_cal_att) #(bz,max_len)
		pos_att_weight = tf.reshape(pos_cal_att_final,[-1, hd_max, 1]) #(bz,max_len,1)
		pos_dm_rexps_hat = tf.reduce_sum(pos_dm_nbrs_vec * pos_att_weight, axis=1) 
		return pos_dm_rexps_hat #(bz,type_dim)

	def _irr_new(self, pos_emb_h, pos_emb_t, pos_emb_r_sm, pos_rel_ss, pos_rel_oo): #(bz,1),...,(bz,1,type_dim)
		pos_h_t_bool = tf.cast(tf.equal(pos_emb_h, pos_emb_t), dtype=tf.float32) #(bz,1), if h==t, ret 1 else 0; 
		pos_h_t_bool_rev = tf.ones_like(pos_h_t_bool)-pos_h_t_bool
		pos_irr_rel = tf.nn.sigmoid(tf.reduce_sum(tf.concat([pos_emb_r_sm, abs(pos_rel_ss-pos_rel_oo)], axis=-1) * self.irr_w, axis=2)) #(bz,1,dim)->(bz,1)
		pos_irr_type = tf.nn.sigmoid(tf.reduce_sum(pos_rel_ss*pos_rel_oo, -1)) #(bz,1,type_dim)->(bz,1)
		pos_irr_pred = pos_irr_rel * pos_irr_type #(bz,1)
		pos_irr_pred = pos_irr_pred * pos_h_t_bool + pos_h_t_bool_rev 
		return pos_irr_pred 

	def _asym_new(self, pos_rel_ss, pos_rel_oo, pos_emb_h_sm, pos_emb_r_sm, pos_emb_t_sm): #(bz,1,dim)
		pos_asym_rc = tf.nn.sigmoid(tf.reduce_sum(pos_rel_oo * pos_rel_ss, -1, keep_dims=False)) #(2721, 1, tp_dim)->(bz,1)
		pos_asym_kge = tf.nn.sigmoid(tf.reduce_sum(self._calc(pos_emb_t_sm, pos_emb_h_sm, pos_emb_r_sm), -1, keep_dims=False)) #(bz,1)
		pos_asym_pred_m = tf.nn.sigmoid(tf.reduce_sum(tf.concat([pos_emb_r_sm], axis=-1) * self.asym_w, axis=2))
		pos_asym_pred = pos_asym_pred_m * pos_asym_kge * pos_asym_rc
		return pos_asym_pred #(bz,1)

	def loss_def(self):
		config = self.get_config()
		#shapes of pos_h are (batch_size, 1)
		pos_emb_h, pos_emb_t, pos_emb_r, pos_y, pos_dm_nbrs, pos_dm_nbrs_len, pos_rg_nbrs, pos_rg_nbrs_len = self.get_positive_instance(in_batch = True)
		neg_emb_h, neg_emb_t, neg_emb_r, neg_y, neg_dm_nbrs, neg_dm_nbrs_len, neg_rg_nbrs, neg_rg_nbrs_len = self.get_negative_instance(in_batch = True)
		
		pos_emb_h_sm = tf.nn.embedding_lookup(self.ent_sm_embeddings, pos_emb_h)
		pos_emb_t_sm = tf.nn.embedding_lookup(self.ent_sm_embeddings, pos_emb_t)
		pos_emb_r_sm = tf.nn.embedding_lookup(self.rel_sm_embeddings, pos_emb_r)

		neg_emb_h_sm = tf.nn.embedding_lookup(self.ent_sm_embeddings, neg_emb_h) 
		neg_emb_t_sm = tf.nn.embedding_lookup(self.ent_sm_embeddings, neg_emb_t)
		neg_emb_r_sm = tf.nn.embedding_lookup(self.rel_sm_embeddings, neg_emb_r)
		# print(pos_emb_h_sm) #(bz,1,dim)

		pos_emb_h_type = tf.nn.embedding_lookup(self.ent_c_embeddings, pos_emb_h)
		pos_emb_t_type = tf.nn.embedding_lookup(self.ent_c_embeddings, pos_emb_t)
		neg_emb_h_type = tf.nn.embedding_lookup(self.ent_c_embeddings, neg_emb_h)
		neg_emb_t_type = tf.nn.embedding_lookup(self.ent_c_embeddings, neg_emb_t)

		#attention for dm
		pos_rel_ss = tf.nn.embedding_lookup(self.rel_ss_embeddings, pos_emb_r) #(bz,1,type_dim)
		pos_dm_rexps_hat = self._aggregation(self.rel_ss_embeddings, pos_rel_ss, pos_dm_nbrs, config.hd_max) #(bz,type_dim)聚合后的关系表示

		neg_rel_ss = tf.nn.embedding_lookup(self.rel_ss_embeddings, neg_emb_r)
		neg_dm_rexps_hat = self._aggregation(self.rel_ss_embeddings, neg_rel_ss, neg_dm_nbrs, config.hd_max)
		
		#attention for rg
		pos_rel_oo = tf.nn.embedding_lookup(self.rel_oo_embeddings, pos_emb_r)
		pos_rg_rexpt_hat = self._aggregation(self.rel_oo_embeddings, pos_rel_oo, pos_rg_nbrs, config.tl_max) 

		neg_rel_oo = tf.nn.embedding_lookup(self.rel_oo_embeddings, neg_emb_r)
		neg_rg_rexpt_hat = self._aggregation(self.rel_oo_embeddings, neg_rel_oo, neg_rg_nbrs, config.tl_max)

		pos_dm_pred = tf.nn.sigmoid(tf.reduce_sum(tf.squeeze(pos_emb_h_type) * pos_dm_rexps_hat, -1, keep_dims=True)) #(bz,1)
		neg_dm_pred = tf.nn.sigmoid(tf.reduce_sum(tf.squeeze(neg_emb_h_type) * neg_dm_rexps_hat, -1, keep_dims=True)) 

		pos_rg_pred = tf.nn.sigmoid(tf.reduce_sum(tf.squeeze(pos_emb_t_type) * pos_rg_rexpt_hat, -1, keep_dims=True)) #(bz,1)
		neg_rg_pred = tf.nn.sigmoid(tf.reduce_sum(tf.squeeze(neg_emb_t_type) * neg_rg_rexpt_hat, -1, keep_dims=True))

		pos_dis_pred = tf.nn.sigmoid(tf.reduce_sum(pos_emb_r_sm * (pos_emb_t_sm - pos_emb_h_sm), -1, keep_dims=False)) #(bz,1)
		neg_dis_pred = tf.nn.sigmoid(tf.reduce_sum(neg_emb_r_sm * (neg_emb_t_sm - neg_emb_h_sm), -1, keep_dims=False))

		pos_irr_pred = self._irr_new(pos_emb_h, pos_emb_t, pos_emb_r_sm, pos_rel_ss, pos_rel_oo) #(bz,1)
		neg_irr_pred = self._irr_new(neg_emb_h, neg_emb_t, neg_emb_r_sm, neg_rel_ss, neg_rel_oo)

		pos_asym_pred = self._asym_new(pos_rel_ss, pos_rel_oo, pos_emb_h_sm, pos_emb_r_sm, pos_emb_t_sm) #(bz,1)
		neg_asym_pred = self._asym_new(neg_rel_ss, neg_rel_oo, neg_emb_h_sm, neg_emb_r_sm, neg_emb_t_sm)
		
		pos_axioms_score = ((1-pos_dm_pred)+(1-pos_rg_pred)+(1-pos_dis_pred)+(1-pos_irr_pred)+(1-pos_asym_pred))
		neg_axioms_score = ((1-neg_dm_pred)+(1-neg_rg_pred)+(1-neg_dis_pred)+(1-neg_irr_pred)+(1-neg_asym_pred))

		pos_emb_score = tf.reduce_sum(self._calc(pos_emb_h_sm, pos_emb_t_sm, pos_emb_r_sm), -1, keep_dims=False) * config.k_rate + pos_axioms_score * config.s_rate #(bz,1)
		neg_emb_score = tf.reduce_sum(self._calc(neg_emb_h_sm, neg_emb_t_sm, neg_emb_r_sm), -1, keep_dims=False) * config.k_rate + neg_axioms_score * config.s_rate
		self.loss = tf.reduce_mean(tf.maximum(pos_emb_score - neg_emb_score + config.margin, 0))
		
	def predict_def(self):
		config = self.get_config()
		predict_emb_h, predict_emb_t, predict_emb_r, predict_dm_nbrs, predict_dm_nbrs_len, predict_rg_nbrs, predict_rg_nbrs_len = self.get_predict_instance()

		predict_emb_h_sm_e = tf.nn.embedding_lookup(self.ent_sm_embeddings, predict_emb_h) 
		predict_emb_t_sm_e = tf.nn.embedding_lookup(self.ent_sm_embeddings, predict_emb_t)
		predict_emb_h_c_e = tf.nn.embedding_lookup(self.ent_c_embeddings, predict_emb_h) 
		predict_emb_t_c_e = tf.nn.embedding_lookup(self.ent_c_embeddings, predict_emb_t)

		predict_emb_r_sm_e = tf.nn.embedding_lookup(self.rel_sm_embeddings, predict_emb_r)
		predict_emb_r_s_e = tf.nn.embedding_lookup(self.rel_ss_embeddings, predict_emb_r)
		predict_emb_r_o_e = tf.nn.embedding_lookup(self.rel_oo_embeddings, predict_emb_r)

		predict_emb_r_s_e = tf.expand_dims(predict_emb_r_s_e,1) #(?,1,type_dim)
		predict_dm_rexps_hat = self._aggregation(self.rel_ss_embeddings, predict_emb_r_s_e, predict_dm_nbrs, config.hd_max) #(?,20)
		self.pred_dm = tf.reshape(tf.nn.sigmoid(tf.cast(tf.reduce_sum(predict_emb_h_c_e * predict_dm_rexps_hat, -1, keep_dims=True), dtype=tf.float32)), [-1]) #(?, 1)->(?,)

		predict_emb_r_o_e = tf.expand_dims(predict_emb_r_o_e,1)
		predict_rg_rexpt_hat = self._aggregation(self.rel_oo_embeddings, predict_emb_r_o_e, predict_rg_nbrs, config.tl_max)
		self.pred_rg = tf.reshape(tf.nn.sigmoid(tf.cast(tf.reduce_sum(predict_emb_t_c_e * predict_rg_rexpt_hat, -1, keep_dims=True), dtype=tf.float32)), [-1]) #(?, )

		predict_emb_h_sm_e = tf.expand_dims(predict_emb_h_sm_e, 1)
		predict_emb_t_sm_e = tf.expand_dims(predict_emb_t_sm_e, 1)
		predict_emb_r_sm_e = tf.expand_dims(predict_emb_r_sm_e, 1)

		self.pred_dis = tf.reshape(tf.nn.sigmoid(tf.reduce_sum(predict_emb_r_sm_e * (predict_emb_t_sm_e - predict_emb_h_sm_e), -1, keep_dims=False)), [-1]) #(?,)

		self.pred_irr = tf.reshape(self._irr_new(tf.expand_dims(predict_emb_h,-1), tf.expand_dims(predict_emb_t,-1), predict_emb_r_sm_e, predict_emb_r_s_e, predict_emb_r_o_e), [-1]) #(?,1)

		self.pred_asym = tf.reshape(self._asym_new(predict_emb_r_s_e, predict_emb_r_o_e, predict_emb_h_sm_e, predict_emb_r_sm_e, predict_emb_t_sm_e), [-1]) #(?,)

		self.pred_emb = tf.reshape(tf.reduce_mean(self._calc(predict_emb_h_sm_e, predict_emb_t_sm_e, predict_emb_r_sm_e), -1, keep_dims=False), [-1]) #(?,)
		self.out_dm = (1-self.pred_dm)
		self.out_rg = (1-self.pred_rg)
		self.out_dis = (1-self.pred_dis)
		self.out_irr = (1-self.pred_irr)
		self.out_asy = (1-self.pred_asym)
		self.pred_emb_and_axiom = config.k_rate * self.pred_emb + config.s_rate * (self.out_dm+self.out_rg+self.out_dis+self.out_irr+self.out_asy)
		
