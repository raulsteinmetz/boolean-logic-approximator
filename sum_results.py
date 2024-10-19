import yaml
from collections import defaultdict

with open('results.yaml', 'r') as file:
    data = yaml.safe_load(file)

precision_data = defaultdict(lambda: defaultdict(list))

for dataset_seed, models in data.items():
    dataset_name = '_'.join(dataset_seed.split('_')[:-1])
    for model, metrics in models.items():
        precision_data[dataset_name][model].append(metrics['precision'])

average_precisions = {}

for dataset, models in precision_data.items():
    average_precisions[dataset] = {}
    for model, precisions in models.items():
        average_precisions[dataset][model] = sum(precisions) / len(precisions)


with open('average_precisions.yaml', 'w') as outfile:
    yaml.dump(average_precisions, outfile, default_flow_style=False)

print("Average precisions have been saved to 'average_precisions.yaml'.")
