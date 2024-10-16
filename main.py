import argparse
import yaml
from util.tools import train_all



def main(config: dict):
    train_all(config)


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