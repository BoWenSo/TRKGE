# TRKGE

This code reproduces results in _Temporal Relevance for Representing Learning over Temporal Knowledge Graphs_

TRKGE is a temporal knowledge graph embedding model that performs link prediction on temporal knowledge graph.
TRKGE is a tensor decomposition model, where the real part learns temporal relevance, and imaginary part learns the relationships between entities, relations and timestamps.


## Datasets

Unzip the datasets, preprocess data by running :
```
python tkbc/process.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

In order to reproduce the results on the smaller datasets in the paper, run the following commands

```
python tkbc/learner.py --dataset ICEWS14 --model TRKGE --rank 2000 --batch_size 500 --emb_reg 1e-2 --time_reg 0.5

python tkbc/learner.py --dataset ICEWS05-15 --model TRKGE --rank 2000 --batch_size 1000 --emb_reg 1e-2 --time_reg 2.5

python tkbc/learner.py --dataset GDELT --model TRKGE --rank 2000  --batch_size 2000 --emb_reg 1e-6 --time_reg 1e-5

```
