# NeuRAN
This repository is the code of paper **Neural Axiom Network for Knowledge Graph Reasoning** submitted to semantic web journal. [paper](https://www.semantic-web-journal.net/content/neural-axiom-network-knowledge-graph-reasoning-0)

## Brief Introduction
---
### Abstract
Knowledge graphs (KGs) generally suffer from incompleteness and incorrectness problems due to the automatic and semi-automatic construction process. Knowledge graph reasoning aims to infer new knowledge or detect noises, which is essential for improving the quality of knowledge graphs. In recent years, various KG reasoning techniques, such as symbolic- and embedding-based methods, have been proposed and shown strong reasoning ability. Symbolic-based reasoning methods infer missing triples according to predefined rules or ontologies. Although rules and axioms have proven to be effective, it is difficult to obtain them. While embedding-based reasoning methods represent entities and relations of a KG as vectors, and complete the KG via vector computation. 
However, they mainly rely on structural information, and ignore implicit axiom information that are not predefined in KGs but can be reflected from data. That is, each correct triple is also a logically consistent triple, and satisfies all axioms. In this paper, we propose a novel **NeuR**al **A**xiom **N**etwork (**NeuRAN**) framework that combines explicit structural and implicit axiom information. It only uses existing triples in KGs without introducing additional ontologies. Specifically, the framework consists of a knowledge graph embedding module that preserves the semantics of triples, and five axiom modules that encode five kinds of implicit axioms using entities and relations in triples. These axioms correspond to five typical object property expression axioms defined in OWL2, including *ObjectPropertyDomain, ObjectPropertyRange, DisjointObjectProperties, IrreflexiveObjectProperty* and *AsymmetricObjectProperty*. The knowledge graph embedding module and axiom modules respectively compute the scores that the triple conforms to the semantics and the corresponding axioms. Evaluations on KG reasoning tasks show the efficiency of our method. Compared with knowledge graph embedding models and CKRL, our method achieves comparable performance on noise detection and triple classification, and achieves significant performance on link prediction. Compared with TransE and TransH, our method improves the link prediction performance on the Hit@1 metric by 22.4% and 21.2% on WN18RR-10\% dataset respectively.

### Model

### Experiments

## Use the Code
---

