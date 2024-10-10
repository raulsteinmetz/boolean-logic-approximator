from tools import learn
from models.mlp import BaseMLP
from data.data_gen import gen




def main():
    ds_path = gen(['not', 'and', 'or', 'if_then'], 5, 10, './data/datasets/', 'ds_10ops_5vars1')
    



if __name__ == '__main__':
    main()