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


@wrap_experiment(archive_launch_repo=False, snapshot_mode="last", use_existing_dir=True, name_parameters='all')
def direct_training(ctxt=None, paired_agent=0):
    args = parser.parse_args()

    set_seed(args.seed)

    env = GarageOvercooked()

    partner = gen_agent(args.partner, env._env, args.partner_load, args)

    env.add_partner_agent(partner)
    env.choose_partner(0)
    print(env._env.partners)

    trainer = Trainer(ctxt)

    hidden_sizes = tuple([args.hidden_size] * args.layer_N)

    policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=hidden_sizes,
                               hidden_nonlinearity=F.relu,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=hidden_sizes,
                                              hidden_nonlinearity=F.relu,
                                              output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=args.lr, eps=args.opti_eps, weight_decay=args.weight_decay)),
                policy,
                max_optimization_epochs=args.ppo_epoch)

    vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=args.critic_lr, eps=args.opti_eps, weight_decay=args.weight_decay)),
                value_function,
                max_optimization_epochs=args.ppo_epoch)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

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
    trainer.train(n_epochs=args.num_env_steps//args.episode_length,
                  batch_size=args.episode_length)


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
direct_training()
