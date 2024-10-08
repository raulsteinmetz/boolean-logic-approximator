import os
import argparse
import torch
import pandas as pd
import numpy as np

def _if_then(p: bool, q: bool):
    return not p or q

def _and(p: bool, q: bool):
    return p and q

def _or(p: bool, q: bool):
    return p or q

def _not(p: bool):
    return not p

op_map = {'not': _not,
          'and': _and,
          'or': _or,
          'if_then': _if_then}

'''
    generates a dataset with a randomized boolean expression,
    number of variables and possible gates are parameterized.
'''

def randomize_ops(ops: list, n_variables: int, n_ops: int):
    nots = np.zeros(n_variables, dtype=bool)
    if 'not' in ops:
        ops.remove('not')
        nots = np.random.choice([True, False], n_variables)

    ops = [np.random.choice(ops) for _ in range(n_ops)]

    def expression(variables: list, nots, ops):
        # compute nots (invert variables in idx where nots are true)
        # compute expression, return value
        pass

def generate(ops: list, n_variables: int, n_ops: int):
    expression = randomize_ops(ops, n_variables, n_ops)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass the list of \
                                     operations and the number of variables.')
    parser.add_argument('--ops', nargs='+', type=str, required=True,
                        help="List of logical operators.")
    parser.add_argument('--n_variables', type=int, required=True,
                        help="Number of variables in the operation.")
    parser.add_argument('--n_ops', type=int, required=True,
                        help="Number of boolean operations performed in the expression.")
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help="Seed for randomizations.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    generate(args.ops, args.n_variables, args.n_ops)

    # use: python3 data/data_gen.py --ops or and not if_then --n_variables 10 --n_ops 20