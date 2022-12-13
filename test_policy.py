import sys

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


def direct_testing(paired_agent=0):
    args = parser.parse_args()

    set_seed(args.seed)

    env = GarageOvercooked()

    partner = gen_agent(args.partner, env._env, args.partner_load, args)

    env.add_partner_agent(partner)
    env.choose_partner(0)
    print(env._env.partners)

    snapshotter = Snapshotter()
    data = snapshotter.load('data/local/experiment/br_training_outer_lr=0.0001_batch_size=7000/')
    policy = data['algo'].policy

    rewards = []

    for _ in range(100):
        steps, max_steps = 0, 200
        done = False
        obs = env.reset()[0]  # The initial observation
        policy.reset()

        full_rew = 0

        while steps < max_steps and not done:
            # print(obs)
            step = env.step(policy.get_action(obs)[0])
            # print(step)
            obs = step.observation
            rew = step.reward
            done = False
            steps += 1
            full_rew += rew
        rewards.append(full_rew)
    print(rewards)
    print(sum(rewards)/len(rewards))
    env.close()


def sb3_main():
    args = parser.parse_args()

    set_seed(args.seed)

    env = SimplifiedOvercooked('random1', 1, False, True)

    partner = gen_agent(args.partner, env, args.partner_load, args)

    env.add_partner_agent(partner, player_num=0)
    # print(env._env.partners)

    ego = stable_baselines3.PPO('MlpPolicy', env, verbose=1)
    ego.learn(total_timesteps=400000)


parser = get_config()
parser.add_argument('partner',
                    choices=EGO_LIST,
                    help='Algorithm for the partner agent')
parser.add_argument('--partner-load',
                    help='File to load the partner agent from')

# sb3_main()
direct_testing()
