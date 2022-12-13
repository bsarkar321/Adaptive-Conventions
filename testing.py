import sys

from collections import Counter

import torch

from statistics import stdev

from math import sqrt

from overcooked_env.overcooked_env import DecentralizedOvercooked

from partner_agents import DecentralizedAgent
from MAPPO.r_actor_critic import R_Actor
from config import get_config

EGO_LIST = ['MAPPO', 'MAML']


def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in
                     sorted(Counter(x).items())])


def generate_gym(args):
    """Generate the gym given the command-line arguments."""
    if args.env_name == "Overcooked":
        args.hanabi_name = "Overcooked"
        return DecentralizedOvercooked(
            args.over_layout,
            ego_agent_idx=0,
            baselines=False,
            use_rew_shape=False
        )
    return None


def gen_agent(value, env, envname, fload, args=None):
    if value == 'MAPPO':
        actor = R_Actor(args, env.observation_space, env.action_space)
        print(fload)
        if fload is None:
            print("NEED TO INPUT FILE")
            sys.exit()
        state_dict = torch.load(fload)
        actor.load_state_dict(state_dict)
        return DecentralizedAgent(actor)


def run_sim(env, ego, alt, N):
    env.add_partner_agent(alt)
    rewards = []
    for game in range(N):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward
        rewards.append(reward)
    print(get_histogram(rewards))
    print(sum(rewards)/len(rewards))
    print("STDEV:", stdev(rewards))
    print("STERR:", stdev(rewards)/sqrt(N))


def main(parser):
    args = parser.parse_args()

    env = generate_gym(args)
    ego = gen_agent(args.ego, env, args.env_name, args.ego_load, args)
    alt = gen_agent(args.partner, env, args.env_name, args.partner_load, args)
    run_sim(env, ego, alt)


if __name__ == '__main__':
    parser = get_config()
    parser.add_argument('ego',
                        choices=EGO_LIST,
                        help='Algorithm for the ego agent')
    parser.add_argument('--ego-load',
                        help='File to load the ego agent from')
    parser.add_argument('partner',
                        choices=EGO_LIST,
                        help='Algorithm for the partner agent')
    parser.add_argument('--partner-load',
                        help='File to load the partner agent from')
    parser.add_argument('-N', type=int, default=500,
                        help="Number of games to test for")
    main(parser)
