# Adaptive-Conventions
Meta-Learning for MARL Adaptation

## Installation
```
conda create --name AdaptiveConventions python=3.10
conda activate AdaptiveConventions
git clone https://github.com/bsarkar321/Adaptive-Conventions

# PantheonRL Submodules
git submodule update --init --recursive
pip install -e .
pip install -e garage/
cd PantheonRL
pip install -e .
pip install -e overcookedgym/human_aware_rl/overcooked_ai
cd ..
```

## Reproducing Results

If you do not want to train from scratch, copy the data folder from [this drive link](https://drive.google.com/drive/folders/1RcYyjHLhfICjdwxs8BJTm6fcsDIi64T4?usp=share_link) into the Adaptive-Conventions folder.

Training the CoMeDi agents:

``` 
python serial_trainer.py --num_env_steps 400000 --pop_size 8 --xp_weight 0.25 --mp_weight 1.0 --lr 2.0e-4 --critic_lr 2.0e-4 --episode_length 4000 --env_length 200 --use_linear_lr_decay --entropy_coef 0.0 --env_name Overcooked --seed 1 --over_layout random1
```

Plotting the cross-play matrix:

```
python score_matrix.py MAPPO --partner-load overcooked_models/1/ --num-train-partners 8
```

Training from scratch (baseline):

```
python garage_training.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --lr 5.0e-5 --critic_lr 5.0e-5 --num_env_steps 200000 --episode_length 200
```

Training the best responses:

```
python best_response.py MAPPO --partner-load overcooked_models/1 --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 7000000 --episode_length 7000 --num-train-partners 2

python best_response.py MAPPO --partner-load overcooked_models/1 --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 7000000 --episode_length 7000 --num-train-partners 4

python best_response.py MAPPO --partner-load overcooked_models/1 --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 7000000 --episode_length 7000 --num-train-partners 7
```

Testing the best responses:

```
python garage_testing.py MAPPO --partner-load overcooked_models/1/ --snapshot-load data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=2 --num-train-partners 8

python garage_testing.py MAPPO --partner-load overcooked_models/1/ --snapshot-load data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=4 --num-train-partners 8

python garage_testing.py MAPPO --partner-load overcooked_models/1/ --snapshot-load data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=7 --num-train-partners 8
```

Few-shot fine tuning (Technique 1):

```
python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=2' --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50 --num-train-partners 2 --use-simple-tuning

python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=4' --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50 --num-train-partners 4 --use-simple-tuning

python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=7' --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50 --num-train-partners 7 --use-simple-tuning
```

Few-shot fine tuning (Technique 2):

```
python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=2' --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50 --num-train-partners 2

python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=4' --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50 --num-train-partners 4

python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=7' --lr 1.0e-4 --critic_lr 1.0e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50 --num-train-partners 7
```

Plot Few-shot results:

```
python plot_results.py
```

MAML:

```
python maml_training.py MAPPO --partner-load overcooked_models/1 --lr 5e-5 --num_env_steps 10000000 --episode_length 10000 --inner-lr 0.01

python maml_training.py MAPPO --partner-load overcooked_models/1 --lr 1e-5 --num_env_steps 10000000 --episode_length 10000 --inner-lr 0.01 --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000_max_num_agents=7'
```

