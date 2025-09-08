# Spurious Backdoors
Backdoors are spurious.

## Necessary Instructions

Please note that the code is able to execute in multiprocessing mode, where the total number of processes can be controlled by changing the ```shots_at_a_time``` variable in the ```p4_discovering_backdoors/config.py``` file.


## Setting up your enviornment
1. Download and install miniconda
```
source install_conda.sh
```

2. After restaring the shell, create a conda environment using u_torch.yml and install all dependencies:
```
source install_u_torch.sh
```

## Running the code
1. Use p4_discovering_backdoors/config.py to set your experiment configurations. All standard configurations are already there (you might need to enable them by uncommenting them in the list named `experimental_setups`).

2. To train your models from scratch, use (this generates a unique name for each model depending on the configuration hyperparameters in `p4_discovering_backdoors/experimental_setups/configurations/model_config.py`.):
```
python _p4a_main.py --train
```

4. Once the models are trained, evaluate and compile results for different threat models:
```
python _p4a_main.py --evaluate --threat_model DC
python _p4a_main.py --evaluate --threat_model MR
python _p4a_main.py --evaluate --threat_model MF
```

5. Once the results are compiled, print results tables (latex format) in the terminal using:
```
python _p4a_main.py --results --threat_model DC
python _p4a_main.py --results --threat_model MR
python _p4a_main.py --results --threat_model MF
```

6. Step 5 will print all the results for SOTA analysis (we will shortly update the code to also save the hyperparameter figures).

## Cite as
```
Blinded
```

