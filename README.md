# transformer-rl

implementation of Decision Transformer on several gridworld environments. This repo is intended to study DT's long-term credit assignment capabilities

[google slides here](https://docs.google.com/presentation/d/1anx2zA1wDQEJZZIEtkFkZA1_X2KHBP3_wz2orBta1eA/edit#slide=id.g27d7f62ef57_0_11)

## Sequence experiment
For the sequence experiment with temporal backward integration: go to `experiments/temporal_backward_integration/sequence_exp_v3.py`

to run this from terminal, go to the root of the project and run:

```
python -m experiments.temporal_backward_integration.sequence_exp_v3
```

Current state of the code:
- onehot position encoding not working yet
- Loss doesn't seem to be converging -- maybe we need to anneal the learning rate

If you want to work in colab / notebook: 
- you can find the Transformer class in `transformer.py` in the root directory
- transformer takes a `config` which has hyperparameters and stuff (optional),
- for the sequence experiment, the config is found in `experiments/temporal_backward_integration/config.py`

all the code uses weights and biases for logging. you might need to change the key for your own wandb account. you can also forgo using it by setting `config.log_to_wandb` to `False`
