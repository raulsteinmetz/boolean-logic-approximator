# MLPs vs. KANs: A Comparative Analysis of Their Capacity to Approximate Boolean Equations

#### Warning

This repository was created for a university presentation. It should not be considered a fully reliable or validated research project. The work was completed in a limited timeframe, and the methodology may not be rigorous or sound.

### Objective

The goal of this project is to compare the performance of Multilayer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs) in approximating boolean equations by analyzing their learning curves.

## Result Examples

<table>
  <tr>
    <td style="border: 1px solid #ddd; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/e14c9524-611e-491e-b51b-3e016f513862" alt="1_2_1" width="400"/>
      <p style="text-align: center;">Comparison of a KAN (gray) and an MLP (green)</p>
    </td>
    <td style="border: 1px solid #ddd; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/109447c7-3ead-4d42-85f1-500371a81f6d" alt="kans" width="400"/>
      <p style="text-align: center;">Comparison of different KAN architecture sizes</p>
    </td>
  </tr>
</table>


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
