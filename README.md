# MLPs vs. KANs: An Analysis of Learning Curves on Boolean Equations

#### Warning

This repository was created for a university presentation. It should not be considered a fully reliable or validated research project. The work was completed in a limited timeframe, and the methodology may not be rigorous or sound.

### Objective

The goal of this project is to compare the performance of Multilayer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs) in approximating boolean equations by analyzing their learning curves.

### Running the experiments
`main.py` generates data, trains and tests models specified in the configuration file.

```
python3 main.py --config_path ./configs/config.yaml
```

### Modify configurations:

To change the configurations (e.g., dataset size, model parameters), edit the `./configs/config.yaml` file.

### Visualizing with TensorBoard

To visualize the learning curves of the models using TensorBoard, run:

```
tensorboard --logdir logs
```

### Repository Structure

```
├── configs/              # Configuration files for model training and experiments
│   └── config.yaml       # Main configuration file
├── data/                 # Dataset storage and code to generate datasets
├── logs/                 # Logs generated during training, used by TensorBoard
├── models/
│   └── mlps/             # MLP class definition with flexible architecture and activation functions
├── util/                 # Training and testing utility functions
├── results.yaml          # Results generated from model testing
├── run.sh                # Script to simplify model training execution
└── tb.sh                 # Script to easily start TensorBoard

```
