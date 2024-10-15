import argparse
import yaml
import os
import numpy as np
from tools import learn
from models.mlp import BaseMLP
from data.data_gen import gen
from data.boolean_dataset import get_loaders


def find_sets(config: dict):
    base_dir = config['dataset']['f_path']
    return [
        os.path.relpath(os.path.join(root, file), base_dir)
        for root, _, files in os.walk(base_dir)
        for file in files if file.endswith('.csv')
    ]


def gen_sets(config: dict):
    ds_paths = [
        gen(
            config['dataset']['ops'],
            n_variables,
            n_ops,
            os.path.join(config['dataset']['f_path'], f'{n_variables}vars_{n_ops}ops'),
            f"{n_variables}_{n_ops}_{seed}",
            seed
        )
        for seed in config['seeds']
        for n_variables in config['dataset']['n_variables']
        for n_ops in config['dataset']['n_ops']
    ]
    return ds_paths


def gen_models(config: dict):
    pass


def main(config: dict):
    ds_paths = gen_sets(config) if not config['dataset']['load'] else find_sets(config)
    print(ds_paths)


def parse_args():
    parser = argparse.ArgumentParser(description="MLP and KAN comparisson")
    parser.add_argument(
        '--config_path', 
        type=str, 
        required=True, 
        help="Path to the YAML file"
    )
    return parser.parse_args()


def yaml_to_dict(yaml_path):
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


if __name__ == '__main__':
    config = yaml_to_dict(parse_args().config_path)
    main(config)