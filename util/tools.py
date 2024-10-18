import os
import torch
import yaml
from kan import *
from torch.nn.functional import selu
from torch.nn import BCEWithLogitsLoss
from data.data_gen import gen
from models.mlp import BaseMLP
from util.seeding import set_all_seeds
from data.boolean_dataset import get_loaders
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def find_sets(config: dict, n_variables: int, seed: int):
    base_dir = config['dataset']['f_path']
    n_vars_str = f'{n_variables}'
    seed_str = f"_{seed}.csv"
    
    return [
        os.path.relpath(os.path.join(root, file), base_dir)
        for root, _, files in os.walk(base_dir)
        for file in files if file.endswith(seed_str) and n_vars_str in file
    ]

def gen_sets(config: dict, n_variables: int, seed):
    ds_paths = [
        gen(
            config['dataset']['ops'],
            n_variables,
            n_ops,
            os.path.join(config['dataset']['f_path'], f'{n_variables}vars_{n_ops}ops'),
            f"{n_variables}_{n_ops}_{seed}",
            seed
        )
        for n_ops in config['dataset']['n_ops']
    ]
    return ds_paths

def gen_mlps(config: dict, input_size: int):
    return [
        BaseMLP(input_size, layer_sizes, selu)
            for layer_sizes in config['models']['mlp']['sizes']
    ]

def gen_kans(config: dict, input_size: int):
    return [
        KAN(
            width=[input_size, width, 1],
            grid=config['models']['kan']['grid'],
            k=config['models']['kan']['k'],
            auto_save=False
        )
        for width in config['models']['kan']['width']
    ]

def summarize_mlp(mlp):
    layers_str = 'x'.join([str(fc.in_features) for fc in mlp.fcs]) + f'x{mlp.fcs[-1].out_features}'
    return f'MLP_{layers_str}'

def summarize_kan(kan):
    return f'KAN_{kan.width[1]}_k{kan.k}_g{kan.grid}'

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.float().view(-1, 1).to(device)
            outputs = model(data)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    precision = correct / total if total > 0 else 0
    return precision

def save_results_to_yaml(results, file_path='results.yaml'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_results = yaml.safe_load(file) or {}
    else:
        existing_results = {}

    for dataset, models in results.items():
        if dataset not in existing_results:
            existing_results[dataset] = {}
        existing_results[dataset].update(models)

    with open(file_path, 'w') as file:
        yaml.dump(existing_results, file)

def train_one(model, device, model_name, ds_path: str, criterion, config: dict):
    log_dir = os.path.join(config['train']['log_dir'] + '/' + os.path.basename(ds_path.replace('.csv', '')), 
                            model_name)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_loader, test_loader = get_loaders(ds_path, config['train']['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    writer = SummaryWriter(log_dir=log_dir)

    model = model.to(device)
    
    loss_moving_avg = None
    smoothing = config['train'].get('smoothing', 0.9)

    progress_bar = tqdm(range(config['train']['n_epochs']), desc='Training', total=config['train']['n_epochs'])

    for epoch in progress_bar:
        running_loss = 0.0
        
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.float().view(-1, 1)
            labels = labels.to(device)
            pred = model.forward(data)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            if loss_moving_avg is None:
                loss_moving_avg = running_loss
            else:
                loss_moving_avg = smoothing * loss_moving_avg + (1 - smoothing) * running_loss

        progress_bar.set_description(f'Epoch {epoch}, Loss: {loss_moving_avg:.4f}')
        writer.add_scalar('Loss/train', loss_moving_avg, epoch)

    test_precision = test_model(model, test_loader, device)
    
    dataset_name = os.path.basename(ds_path).replace('.csv', '')
    results = {dataset_name: {model_name: {'precision': test_precision}}}
    save_results_to_yaml(results)

    writer.close()

def train_all(config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('--------------------')
    print('STARTING EXPERIMENTS')
    print('--------------------', end='\n\n')
    
    for seed in config['seeds']:
        set_all_seeds(seed)
        for n_variables in config['dataset']['n_variables']:
            ds_paths = gen_sets(config, n_variables, seed) \
                if not config['dataset']['load'] else find_sets(config, n_variables, seed)
            
            for ds_path in ds_paths:
                for kan in gen_kans(config, n_variables):
                    model_name = summarize_kan(kan)
                    print(f'DATASET: {ds_path}, MODEL: {model_name}')
                    train_one(kan, device, model_name, os.path.join(config['dataset']['f_path'], ds_path), BCEWithLogitsLoss(), config)

                for mlp in gen_mlps(config, n_variables):
                    model_name = summarize_mlp(mlp)
                    print(f'DATASET: {ds_path}, MODEL: {model_name}')
                    train_one(mlp, device, model_name, os.path.join(config['dataset']['f_path'], ds_path), BCEWithLogitsLoss(), config)
