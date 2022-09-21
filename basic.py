import config
import models
import tensorflow as tf
import numpy as np
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']='0'
#Input training files from benchmarks/FB15K/ folder.
#tf.set_random_seed(0)
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("./data/FB15K237_N1/")
con.set_data_dir("./data/FB15K237_N1/")
con.set_neg_path("neg_10%.txt")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_test_noise_detection(True)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(1.0)
con.set_margin(4.0)
con.set_bern(1)
con.set_opt_method("SGD")
con.set_kgerate(1)
con.set_scorerate(0.05)
con.set_tp_dimension(20)
con.set_sm_dimension(200)

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.NbrAttention_transe)
# con.set_model(models.NbrAttention_transh)

#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()

