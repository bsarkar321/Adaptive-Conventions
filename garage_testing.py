import sys
import os

from config import get_config

import torch

from torch.nn import functional as F

from MAPPO.r_actor_critic import R_Actor

from partner_agents import FixedMAPPOAgent

from overcooked_env.garage_overcooked import GarageOvercooked

from discrete_mlp_policy import DiscreteMLPPolicy

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from garage.experiment import MetaEvaluator

from custom_maml import CustomMAML

EGO_LIST = ['MAPPO', 'MAML']

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from garage.experiment import Snapshotter
from garage.sampler.utils import rollout



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

def run_sim(env, ego, alt, N):
    env.choose_partner(alt)
    rewards = []
    for game in range(N):
        rewards.append(sum(rollout(env, ego)['rewards']))
    # print(get_histogram(rewards))
    # print(sum(rewards)/len(rewards))
    # print("STDEV:", stdev(rewards))
    # print("STERR:", stdev(rewards)/sqrt(N))
    return sum(rewards)/len(rewards)

def print_score_matrix(max_num_agents):
    set_seed(args.seed)

    snapshotter = Snapshotter()
    data = snapshotter.load(args.snapshot_load)
    policy = data['algo'].policy

    env = GarageOvercooked()

    partner_list_locs = os.listdir(args.partner_load)
    partner_list_locs = [x for x in partner_list_locs if os.path.isdir(os.path.join(args.partner_load, x))]
    num_agents = 0
    
    for f in sorted(partner_list_locs, key=lambda x: int(x[10:])):
        new_base = os.path.join(args.partner_load, f)
        if not os.path.isdir(new_base):
            continue
        new_load = os.path.join(new_base, "models/actor.pt")
        partner = gen_agent(args.partner, env._env, new_load, args)

        if num_agents < max_num_agents:
            env.add_partner_agent(partner)
            num_agents += 1
        else:
            break
    env.choose_partner(0)
    print(env._env.partners)
    print(env.get_all_partners())
    # partner_copy = env.get_all_partners()
    rewards = []
    for i in range(max_num_agents):
        rewards.append([])
        rew = run_sim(env, policy, i, 100)
        rewards[-1].append(rew)
        if i == max_num_agents - 1:
            print(f"{rew} \\\\")
        else:
            print(rew, end=" & ")

    rewards = np.array(rewards)
    print(rewards)

parser = get_config()
parser.add_argument('partner',
                    choices=EGO_LIST,
                    help='Algorithm for the partner agent')
parser.add_argument('--partner-load',
                    help='Directory to load the partner agents from')
parser.add_argument('--snapshot-load',
                    help='Snapshot to load best response from')
parser.add_argument('--num-train-partners',
                    type=int,
                    help='Number of partners to train with')

args = parser.parse_args()

print_score_matrix(max_num_agents=args.num_train_partners)
