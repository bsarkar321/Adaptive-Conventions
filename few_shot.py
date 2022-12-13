import sys
import copy

from config import get_config

import torch

from torch.nn import functional as F

from MAPPO.r_actor_critic import R_Actor

from partner_agents import FixedMAPPOAgent

from overcooked_env.garage_overcooked import GarageOvercooked

from overcooked_env.overcooked_env import SimplifiedOvercooked

from discrete_mlp_policy import DiscreteMLPPolicy

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler, LocalSampler
from garage.torch.algos import PPO
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from garage.sampler.utils import rollout
from garage.experiment import Snapshotter

import stable_baselines3

from statistics import stdev

from math import sqrt, ceil

from garage.sampler.default_worker import DefaultWorker


from garage import EpisodeBatch

from garage.torch import update_module_params

import numpy as np


EGO_LIST = ['MAPPO', 'MAML']


def gen_agent(value, env, fload, args=None):
    if value == 'MAPPO':
        actor = R_Actor(args, env.observation_space, env.action_space)
        print(fload)
        if fload is None:
            print("NEED TO INPUT FILE")
            sys.exit()
        state_dict = torch.load(fload)
        actor.load_state_dict(state_dict)
        return FixedMAPPOAgent(actor, env)


def finetuning(ctxt, args, num, policy, value_function):
    # policy_params = dict(policy.named_parameters())
    # value_params = dict(value_function.named_parameters())
    # cloned_old_policy = copy.deepcopy(policy)
    # cloned_policy = copy.deepcopy(policy)
    # cloned_value = copy.deepcopy(value_function)
    set_seed(args.seed + num)
    
    policy_params = copy.deepcopy(policy.state_dict())
    value_params = copy.deepcopy(value_function.state_dict())
    # print(value_params)

    env = GarageOvercooked()

    partner = gen_agent(args.partner, env._env, args.partner_load, args)

    env.add_partner_agent(partner)
    env.choose_partner(0)
    print(env._env.partners)

    trainer = Trainer(ctxt)

    policy_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=args.lr, eps=args.opti_eps, weight_decay=args.weight_decay)),
                policy,
                max_optimization_epochs=args.ppo_epoch)

    vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=args.critic_lr, eps=args.opti_eps, weight_decay=args.weight_decay)),
                value_function,
                max_optimization_epochs=args.ppo_epoch)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           n_workers=1)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               policy_optimizer=policy_optimizer,
               vf_optimizer=vf_optimizer,
               discount=args.gamma,
               gae_lambda=args.gae_lambda,
               center_adv=False)

    trainer.setup(algo, env)
    ftsteps = 0

    # rewards = [sum(rollout(env, policy)['rewards'])]
    episodes = []
    rewards = []
    for ftsteps in range(args.num_env_steps//args.episode_length):
        curepisode = sampler._workers[0].rollout()
        episodes.append(curepisode)
        rewards.append(sum(curepisode.rewards))
        if ftsteps == args.num_env_steps//args.episode_length - 1:
            break
        
        if args.use_simple_tuning:
            algo._train_once(ftsteps, curepisode)
        else:
            algo._old_policy.load_state_dict(policy_params)
            algo.policy.load_state_dict(policy_params)
            algo._value_function.load_state_dict(value_params)
            # algo._train_once(ftsteps, curepisode)

            # policy_optimizer._max_optimization_epochs = ceil(args.ppo_epoch / (ftsteps + 1))
            # vf_optimizer._max_optimization_epochs = ceil(args.ppo_epoch / (ftsteps + 1))
            for i in range(ftsteps + 1):
                algo._train_once(ftsteps, EpisodeBatch.concatenate(*episodes))
            # algo._train_once(ftsteps, episodes[i])
    # evaluate(algo.policy, partner, ftsteps)
    print("reward trajectory:", rewards)
    # update_module_params(policy, policy_params)
    # update_module_params(value_function, value_params)
    return rewards

@wrap_experiment(archive_launch_repo=False, snapshot_mode="last", use_existing_dir=True, name_parameters='all')
def multi_finetuning(ctxt=None, iters=500):
    rewards = []
    args = parser.parse_args()
    snapshotter = Snapshotter()
    data = snapshotter.load(args.snapshot_load)
    policy = data['algo'].policy
    value_function = data['algo']._value_function
    for i in range(iters):
        rewards.append(finetuning(ctxt, args, i, copy.deepcopy(policy), copy.deepcopy(value_function)))
        np.save('few_shot_results' + ('_simple' if args.use_simple_tuning else '') + args.num_train_partners, np.array(rewards))
    print(rewards)
    print("Human-readable:")
    for reward in rewards:
        print(reward)
    


parser = get_config()
parser.add_argument('partner',
                    choices=EGO_LIST,
                    help='Algorithm for the partner agent')
parser.add_argument('--partner-load',
                    help='File to load the partner agent from')

parser.add_argument('--snapshot-load',
                    help='Snapshot to load best response from')
parser.add_argument('--use-simple-tuning',
                    action='store_true', default=False,
                    help='Use simple few-shot learning')
parser.add_argument('--num-train-partners',
                    help='Number of partners to train with')

# 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000/'

# python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --lr 1.0e-4 --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000/' --critic_lr 1.0e-4 --num_env_steps 2000 --episode_length 200 --ppo_epoch 100

# python few_shot.py MAPPO --partner-load overcooked_models/1/convention7/models/actor.pt --snapshot-load 'data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000/' --lr 1.2e-4 --critic_lr 1.2e-4 --num_env_steps 4000 --episode_length 200 --ppo_epoch 50

# sb3_main()
multi_finetuning()
