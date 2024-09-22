### TAUgrok

A pytorch reproduction of the paper [GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS](https://arxiv.org/pdf/2201.02177) by Power et al.

Project setup - 
1. Clone the directory
2. Run ```pip install -r requirements.txt```

### Grok cli - Run the project

| Argument         | Type     | Default   | Description                           |
|------------------|----------|-----------|---------------------------------------|
| `--lr`           | `float`  | `1e-5`    | Learning rate for the model           |
| `--seed`         | `int`    | `42`      | Random seed for reproducibility       |
| `--optim_steps`  | `int`    | `100000`  | Number of optimization steps          |
| `--save-models`  | `bool`   | `False`   | Flag to save model checkpoints        |
| `--save-metrics` | `bool`   | `False`   | Flag to save performance metrics      |
| `--operation`    | `str`    | `"div"`   | Operation dataset, supports 'div' or 'add' |

Run example
```python grok.py --lr=1e-4 --save-models --save-metrics --optim_steps=100000 --operation=add```

If save-models or save-metrics flags are set to True, a directory named ```{operation}_lr_{lr}_seed_{seed}_optim_steps_{optim_steps}```
will be created automatically, and the artifacts will be saved there.


### Figures

To show the figures, the model must be run with both save-metrics and save-models set to True at leat once.
You can also use the examples in the repository if running the model is not possible.

To run the figures, open one of the ```*_figures.ipynb``` files, replace the parameters in the second cell to the ones
you would like to view figures for, and run the full notebook.

Example figure from running the example line in the cli section:

![image](/assets/div_lr_0.0001_seed_42_optim_steps_100000.png)


### Technical details
The projects runs smoothly on machines with GPUs (reproduced on RTX 4090), it was not checked on machines with CPU only.

The avg GPU consumption for the modular division dataset and default parameters mentioned in the paper is 1.5GB, and the run time for 10^5 optimisation steps is ~15 minutes 

> If wandb is installed on the machine running the code, a new project will be opened ('grokking') and the metrics will be logged automatically.
