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

op_map = {'and': _and,
          'or': _or,
          'if_then': _if_then}

'''
    generates a dataset with a randomized boolean expression,
    number of variables and possible gates are parameterized.
'''

def create_exp(ops: list, n_variables: int, n_ops: int):
    nots = np.zeros(n_ops, dtype=bool)
    if 'not' in ops:
        ops.remove('not')
        nots = np.random.choice([True, False], n_ops)
    
    ops = [np.random.choice(ops) for _ in range(n_ops)]
    order_of_variables = [np.random.choice(range(n_variables)) for _ in range(n_ops)]

    def exp_f(x: list):
        variables = [x[i] for i in order_of_variables]
        variables = [not v if n else v for v, n in zip(variables, nots)]
        value = variables[0]
        for i in range(1, n_ops):
            value = op_map[ops[i-1]](value, variables[i])
        return value

    exp_s = f"v{order_of_variables[0]}" if nots[0] == False else f"~v{order_of_variables[0]}"
    for i in range(1, n_ops):
        exp_s += f" {ops[i-1]} " + (f"~v{order_of_variables[i]}" if nots[i] else f"v{order_of_variables[i]}")

    return exp_f, exp_s

def generate(ops: list, n_variables: int, n_ops: int):
    expression, exp_s = create_exp(ops, n_variables, n_ops)
    

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