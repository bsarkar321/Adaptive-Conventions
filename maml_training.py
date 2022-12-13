import sys
import os

from config import get_config

import torch

from torch.nn import functional as F

from MAPPO.r_actor_critic import R_Actor

from partner_agents import FixedMAPPOAgent

from overcooked_env.garage_overcooked import GarageOvercooked, OvercookedTaskSampler

from discrete_mlp_policy import DiscreteMLPPolicy

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import MAMLPPO
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from garage.experiment import MetaEvaluator

from custom_maml import CustomMAML

from garage.experiment import Snapshotter

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
def maml_training(ctxt=None, inner_lr=0.001, outer_lr=1e-4, batch_size=1000):
    set_seed(args.seed)

    env = GarageOvercooked()

    partner_list_locs = os.listdir(args.partner_load)
    partner_list_locs = [x for x in partner_list_locs if os.path.isdir(os.path.join(args.partner_load, x))]

    for f in sorted(partner_list_locs, key=lambda x: int(x[10:])):
        new_base = os.path.join(args.partner_load, f)
        if not os.path.isdir(new_base):
            continue
        new_load = os.path.join(new_base, "models/actor.pt")
        partner = gen_agent(args.partner, env._env, new_load, args)

        env.add_partner_agent(partner)
    env.choose_partner(0)
    print(env._env.partners)
    print(env.get_all_partners())
    partner_copy = env.get_all_partners()

    trainer = Trainer(ctxt)

    hidden_sizes = tuple([64] * args.layer_N)

    if args.snapshot_load is not None:
        snapshotter = Snapshotter()
        data = snapshotter.load(args.snapshot_load)
        policy = data['algo'].policy
        value_function = data['algo']._value_function
    else:
        policy = DiscreteMLPPolicy(env.spec,
                                   hidden_sizes=hidden_sizes,
                                   hidden_nonlinearity=F.relu,
                                   output_nonlinearity=None)

        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                  hidden_sizes=hidden_sizes,
                                                  hidden_nonlinearity=F.relu,
                                                  output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    train_idxs = [0, 1, 2, 3, 4, 5, 6]

    train_sampler = OvercookedTaskSampler(train_idxs, partner_copy)
    # print(task_sampler.n_tasks)

    test_idxs = [7]

    test_sampler = OvercookedTaskSampler(test_idxs, partner_copy)

    meta_eval = MetaEvaluator(
        test_task_sampler=test_sampler,
        n_test_tasks=5,
        n_exploration_eps=1,
        n_test_episodes=1,
    )

    algo = CustomMAML(env=env,
                      policy=policy,
                      value_function=value_function,
                      sampler=sampler,
                      task_sampler=train_sampler,
                      inner_lr=args.inner_lr,
                      outer_lr=args.lr,
                      discount=args.gamma,
                      gae_lambda=1.0,
                      center_adv=False,
                      meta_batch_size=len(train_idxs),
                      num_grad_updates=1,
                      meta_evaluator=meta_eval,
                      evaluate_every_n_epochs=1,
                      max_optimization_epochs=args.ppo_epoch,
                      total_iters=args.num_env_steps//args.episode_length)
    # algo = MAMLPPO(env=env,
    #                policy=policy,
    #                value_function=value_function,
    #                sampler=sampler,
    #                task_sampler=train_sampler,
    #                inner_lr=args.inner_lr,
    #                outer_lr=args.lr,
    #                discount=args.gamma,
    #                gae_lambda=1.0,
    #                center_adv=False,
    #                meta_batch_size=train_sampler.n_tasks,
    #                num_grad_updates=1,
    #                meta_evaluator=meta_eval,
    #                evaluate_every_n_epochs=1)
    trainer.setup(algo, env)
    trainer.train(n_epochs=args.num_env_steps//args.episode_length,
                  batch_size=args.episode_length)


parser = get_config()
parser.add_argument('partner',
                    choices=EGO_LIST,
                    help='Algorithm for the partner agent')
parser.add_argument('--partner-load',
                    help='Directory to load the partner agents from')
parser.add_argument('--inner-lr', type=float, default=0.01,
                    help='Inner lr for MAML')
parser.add_argument('--snapshot-load',
                    help='Snapshot to load best response from')

args = parser.parse_args()

maml_training(inner_lr=args.inner_lr, outer_lr=args.lr, batch_size=args.episode_length)
