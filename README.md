# NeuRAN
This repository is the code of paper [**Neural Axiom Network for Knowledge Graph Reasoning**](https://www.semantic-web-journal.net/content/neural-axiom-network-knowledge-graph-reasoning-0) submitted to semantic web journal.  
In this paper, we propose a novel **NeuR**al **A**xiom **N**etwork (**NeuRAN**) framework that combines explicit structural and implicit axiom information. The framework consists of a knowledge graph embedding module that preserves the semantics of triples, and five axiom modules that encode five kinds of implicit axioms using entities and relations in triples. These axioms correspond to five typical object property expression axioms defined in OWL2, including *ObjectPropertyDomain, ObjectPropertyRange, DisjointObjectProperties, IrreflexiveObjectProperty* and *AsymmetricObjectProperty*. The knowledge graph embedding module and axiom modules respectively compute the scores that the triple conforms to the semantics and the corresponding axioms. 

![image](https://github.com/JuanLi1621/NeuRAN/blob/main/swj_frw.png)


## Requirements
python 3.6.13  
tensorflow 1.14.0  
tensorflow-gpu 1.14.0  
cudatoolkit 10.1  
cudnn7.6.5

## Data
For each dataset, we generate noisy triples with the ratio to be 10%, 20%, and 40% of the positive triples, and the noisy triples are added to the training set as part of the training triples. Take FB15K237_N1 as an example.  
+ entity2id.txt: The first line is the number of entities, then each line corresponds to (entity name, entity id).  
+ relation2id.txt: The first line is the number of relations, then each line corresponds to (relation name, relation id).  
+ train2id.txt: The first line is the number of triples for training, containing positive triples of FB15K237 and the generated noisy triples. Then each line corresponds to (head entity, tail entity, relation).  
+ valid2id.txt: The first line is the number of triples for validating. Then each line corresponds to (head entity, tail entity, relation).  
+ test2id.txt: The first line is the number of triples for testing. Then each line corresponds to (head entity, tail entity, relation).  
+ valid2id_neg.txt: The negative triples generated from positive triples of validating file/valid2id.txt.  
+ test2id_neg.txt: The negative  triples generated from positive triples of testing file/test2id.txt.  
+ train2id_origin.txt: Positive triples of FB15K237.  
+ neg_10%.txt: The generated noisy triples.
+ ori_att_h_out_ranked.txt: The relations connected by the head entities.   
+ ori_att_t_in_ranked.txt: The relations connected to the tail entities.  

## Training and Testing
1. Install TensorFlow
2. Clone this repository
3. Complile C++ files  
```
$ bash make.sh
```
4. Training and testing. 
```
python basic.py
```
In basic.py, we set configure parameters for training, including dataset, noisy triple path, the data traversing rounds, batch size, learning rate, margin, bern sampling, optimization algorithm, rates of kg embedding score and axiom score, dimensions of type and semantic embeddings, and the selected knowledge graph embedding model.
```
import config
import models
import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()

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

# Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
# Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
# Initialize experimental settings.
con.init()
# Set the knowledge embedding model.
con.set_model(models.NbrAttention_transe)
# con.set_model(models.NbrAttention_transh)

# Train the model.
con.run()
# To test models after training needs "set_test_flag(True)".
con.test()
```
We conduct experiments on link prediction, triple classification and noise detection tasks.

### Acknowledgement
We refer to the code of [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0). Thanks for their contributions.
