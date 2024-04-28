## Datasets

Unzip the datasets,  add them to the package data folder by running :
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