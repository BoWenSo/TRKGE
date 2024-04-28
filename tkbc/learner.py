# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import pickle
from typing import Dict
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, TRKGE
from regularizers import  Lambda3,  N3, N4, Lambda3_two

import time


parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', default = "ICEWS14", type=str,
    help="Dataset name"
)
models = [
    'TTransE','ComplEx', 'TComplEx', 'TNTComplEx', 'TRKGE'
]
parser.add_argument(
    '--model', default="TRKGE", choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=10, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=500, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0.01, type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=1, type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb',  action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--base_time', default=False, action="store_true"
)


args = parser.parse_args()

dataset = TemporalDataset(args.dataset)


sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TRKGE': TRKGE(sizes, args.rank, False),
}[args.model]

model = model.cuda()


opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

if args.model == 'TRKGE':
    emb_reg = N4(args.emb_reg)
else:
    emb_reg = N3(args.emb_reg)

time_reg = Lambda3(args.time_reg)


time_start = time.time()
for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}


    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)

        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            print("Epoch:",epoch)
            print("valid: ", valid['MRR'], 'hits@[1,3,10]:', valid['hits@[1,3,10]'])
            print("test: ", test['MRR'], 'hits@[1,3,10]:', test['hits@[1,3,10]'])
            print("train: ", train['MRR'], 'hits@[1,3,10]:', train['hits@[1,3,10]'])
        #
        #     if valid['MRR'] > max_test_mrr:
        #         max_test_mrr = valid['MRR']
        #         hit = valid['hits@[1,3,10]']
        #
        #         file = "./logs/log_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(args.model, args.dataset, args.rank,
        #                                                                              args.learning_rate, args.batch_size,
        #                                                                              args.emb_reg, args.time_reg, args.cycle)
        #
        #         with open(file, 'w') as f:
        #             f.write(str(args) + '\n')
        #             f.write('Hit@1:%f\nHit@3:%f\nHit@10:%f\nMRR:%f\nEpoch:%f' % (
        #             hit[0], hit[1], hit[2], max_test_mrr,epoch))
        #
        #         print("saved model!")
        #         torch.save(model, "./logs/log_{}_{}_{}_{}_{}_{}_{}.pkl".format(args.model, args.dataset, args.rank,
        #                                                                              args.learning_rate, args.batch_size,
        #                                                                              args.emb_reg, args.time_reg))
        #
        # time_end = time.time()
        # time_sum = time_end - time_start
        # print(time_sum)

