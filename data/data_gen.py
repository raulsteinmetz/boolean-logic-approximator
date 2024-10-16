import os
import pandas as pd
import numpy as np


def _if_then(p: bool, q: bool):
    return not p or q

def _and(p: bool, q: bool):
    return p and q

def _or(p: bool, q: bool):
    return p or q

op_map = {
    'and': _and,
    'or': _or,
    'if_then': _if_then
}

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
        return int(value)


    exp_s = f"v{order_of_variables[0]}" if nots[0] == False else f"~v{order_of_variables[0]}"
    for i in range(1, n_ops):
        exp_s += f" {ops[i-1]} " + (f"~v{order_of_variables[i]}" if nots[i] else f"v{order_of_variables[i]}")

    return exp_f, exp_s

def gen(ops: list, n_vars: int, n_ops: int, f_path: str, f_name:str):
    exp_f, exp_s = create_exp(ops, n_vars, n_ops)
    
    combinations = np.array([list(map(int, np.binary_repr(i, width=n_vars))) for i in range(2**n_vars)])
    results = [exp_f(values) for values in combinations]
    
    df = pd.DataFrame(combinations, columns=[f'v{i}' for i in range(n_vars)])
    df['result'] = results
    
    os.makedirs(f_path, exist_ok=True)
    
    output_file = os.path.join(f_path, f'{f_name}.csv' if f_name else f'{exp_s}.csv')
    df.to_csv(output_file, index=False)

    return output_file
