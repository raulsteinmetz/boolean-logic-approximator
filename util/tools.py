import os
from torch.nn.functional import relu, sigmoid, tanh
from data.data_gen import gen
from models.mlp import BaseMLP
from util.seeding import set_all_seeds
from data.boolean_dataset import get_loaders


def find_sets(config: dict, seed: int):
    base_dir = config['dataset']['f_path']
    seed_str = f"_{seed}.csv"
    
    return [
        os.path.relpath(os.path.join(root, file), base_dir)
        for root, _, files in os.walk(base_dir)
        for file in files if file.endswith(seed_str)
    ]


def gen_sets(config: dict, seed):
    ds_paths = [
        gen(
            config['dataset']['ops'],
            n_variables,
            n_ops,
            os.path.join(config['dataset']['f_path'], f'{n_variables}vars_{n_ops}ops'),
            f"{n_variables}_{n_ops}_{seed}",
            seed
        )
        for n_variables in config['dataset']['n_variables']
        for n_ops in config['dataset']['n_ops']
    ]
    return ds_paths


def gen_mlps(config: dict):
    activ_func_map = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh
    }
    mlp_dict = {}
    for input_size in config['dataset']['n_variables']:
        mlp_dict[input_size] = [
            BaseMLP(input_size, layer_sizes, activ_func_map[activ_func])
            for layer_sizes in config['models']['mlp']['sizes']
            for activ_func in config['models']['mlp']['activ_functions']
        ]
    
    return mlp_dict


def train_one(model, ds_path: str, config: dict):  
    train_loader, test_loader = get_loaders(ds_path, config['train']['batch_size'])

    for epoch in range(config['train']['n_epochs']):
        for data, labels in train_loader:
            pred = model.forward(data)
            print(pred)
            print(labels)
            exit()


def train_all(config: dict):
    for seed in config['seeds']:
        set_all_seeds(seed)
        ds_paths = gen_sets(config, seed) if not config['dataset']['load'] else find_sets(config, seed)
        mlps = gen_mlps(config)

        [
            train_one(mlp, os.path.join(config['dataset']['f_path'], ds_path), config)
            for input_size in config['dataset']['n_variables']
            for mlp in mlps[input_size]
            for ds_path in ds_paths
        ]

