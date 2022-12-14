from .utils.util import _t2n
from .rMAPPOPolicy import R_MAPPOPolicy
from .r_mappo import R_MAPPO
from .utils.shared_buffer import SharedReplayBuffer

import os
import time
from collections import Counter

import numpy as np
import torch


def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])



class MainPlayer:
    def __init__(self, config):
        self._init_vars(config)
        share_observation_space = self.env.share_observation_space

        # policy network
        self.policy = R_MAPPOPolicy(
            self.all_args,
            self.env.observation_space,
            share_observation_space,
            self.env.action_space,
            device=self.device,
        )

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = R_MAPPO(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.env.observation_space,
            share_observation_space,
            self.env.action_space,
        )

    def _init_vars(self, config):
        self.all_args = config["all_args"]
        self.env = config["env"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save_dir = str(self.run_dir / "models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.true_total_num_steps = 0
        self.ego_id = 0

    def collect_episode(self, buffer=None, length=None, save_scores=True, turn_based=True):
        buffer = buffer or self.buffer
        self.use_obs, self.use_share_obs, self.use_available_actions = self.env.reset()
        self.running_score = 0
        if length is None:
            length = self.episode_length
        if save_scores:
            self.scores = []
        for _ in range(length):
            # Sample actions
            done = self.next_step(self.scores if save_scores else None)

            # insert turn data into buffer
            if turn_based:
                buffer.chooseinsert(
                    self.turn_share_obs,
                    self.turn_obs,
                    self.turn_rnn_states,
                    self.turn_rnn_states_critic,
                    self.turn_actions,
                    self.turn_action_log_probs,
                    self.turn_values,
                    self.turn_rewards,
                    self.turn_masks,
                    self.turn_bad_masks,
                    self.turn_active_masks,
                    self.turn_available_actions,
                )
            else:
                buffer.insert(
                    self.turn_share_obs,
                    self.turn_obs,
                    self.turn_rnn_states,
                    self.turn_rnn_states_critic,
                    self.turn_actions,
                    self.turn_action_log_probs,
                    self.turn_values,
                    self.turn_rewards,
                    self.turn_masks,
                    self.turn_bad_masks,
                    self.turn_active_masks,
                    self.turn_available_actions,
                )
            # env reset
            if done:
                (
                    self.use_obs,
                    self.use_share_obs,
                    self.use_available_actions,
                ) = self.env.reset()

    def log(self, train_infos, episode, episodes, total_num_steps, start):
        # save model
        if episode % self.save_interval == 0 or episode == episodes - 1:
            self.save()

        if episode == 0:
            # Setup files
            files = []
            # log.txt
            # Env algo exp updates ... avg score, avg xp score
            files.append(self.log_dir + "/log.txt")

            # sp.txt
            # t: episode, Counter
            files.append(self.log_dir + "/sp.txt")

            os.makedirs(self.log_dir, exist_ok=True)
            for file in files:
                with open(file, "w", encoding="UTF-8"):
                    pass

        # log information
        if train_infos is not None or (
            episode % self.log_interval == 0 and episode > 0
        ):
            end = time.time()
            files = {}

            average_score = np.mean(self.scores)

            general_log = (
                f"Updates:{episode}/{episodes},"
                + f"Timesteps:{total_num_steps}/{self.num_env_steps},"
                + f"FPS:{total_num_steps//(end-start)},"
                + f"avg_sp:{average_score}"
            )

            print(
                "\n Env {} Algo {} Exp {} updates {}/{} episodes, \
                total num timesteps {}/{}, FPS {}.\n".format(
                    self.all_args.hanabi_name,
                    self.algorithm_name,
                    self.experiment_name,
                    episode,
                    episodes,
                    total_num_steps,
                    self.num_env_steps,
                    int(total_num_steps / (end - start)),
                )
            )

            print("average score is {}.".format(average_score))

            train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

            # self.log_train(train_infos, self.true_total_num_steps)
            print(train_infos)
            general_log += "," + ",".join(
                [f"{key}:{val}" for key, val in train_infos.items()]
            )

            files["log.txt"] = general_log

            files["sp.txt"] = get_histogram(self.scores)
            print("Self-play Scores counts: ", files["sp.txt"])

            for key, val in files.items():
                with open(f"{self.log_dir}/{key}", "a", encoding="UTF-8") as file:
                    file.write(f"episode:{episode},{val}\n")
                
    # def log(self, train_infos, episode, episodes, total_num_steps, start):
    #     # save model
    #     if episode % self.save_interval == 0 or episode == episodes - 1:
    #         self.save()

    #     # log information
    #     if train_infos is not None or (
    #         episode % self.log_interval == 0 and episode > 0
    #     ):
    #         end = time.time()
    #         print(
    #             "\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
    #                 self.all_args.hanabi_name,
    #                 self.algorithm_name,
    #                 self.experiment_name,
    #                 episode,
    #                 episodes,
    #                 total_num_steps,
    #                 self.num_env_steps,
    #                 int(total_num_steps / (end - start)),
    #             )
    #         )

    #         average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
    #         print("average score is {}.".format(average_score))
    #         train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

    #         # self.log_train(train_infos, self.true_total_num_steps)
    #         print(train_infos)
    #         print("Scores counts:", sorted(Counter(self.scores).items()))

    def run(self):
        self.setup_data()
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        train_infos = None
        total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.collect_episode()
            total_num_steps += self.episode_length
            # post process
            self.log(train_infos, episode, episodes, total_num_steps, start)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            print("DONE TRAINING:", episode)

    def next_step(self, scores=None):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_state,
            rnn_state_critic,
        ) = self.trainer.policy.get_actions(
            self.use_share_obs,
            self.use_obs,
            self.turn_rnn_states[0, self.ego_id],
            self.turn_rnn_states_critic[0, self.ego_id],
            self.turn_masks[0, self.ego_id],
            self.use_available_actions,
        )
        self.turn_obs[0, self.ego_id] = self.use_obs.copy()
        self.turn_share_obs[0, self.ego_id] = self.use_share_obs.copy()
        self.turn_available_actions[0, self.ego_id] = self.use_available_actions.copy()
        self.turn_values[0, self.ego_id] = _t2n(value)
        self.turn_actions[0, self.ego_id] = _t2n(action)
        env_actions = _t2n(action)
        self.turn_action_log_probs[0, self.ego_id] = _t2n(action_log_prob)
        self.turn_rnn_states[0, self.ego_id] = _t2n(rnn_state)
        self.turn_rnn_states_critic[0, self.ego_id] = _t2n(rnn_state_critic)
        self.turn_active_masks[0, 1 - self.ego_id] = 0
        self.turn_active_masks[0, self.ego_id] = 1
        (obs, share_obs, available_actions), rewards, done, info = self.env.step(
            env_actions
        )
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()
        self.turn_rewards[0, self.ego_id] = rewards

        self.running_score += rewards

        if done:
            self.turn_masks[0, self.ego_id] = 0
            self.turn_rnn_states[0, self.ego_id] = 0
            self.turn_rnn_states_critic[0, self.ego_id] = 0

            if scores is not None:
                scores.append(self.running_score)

            self.running_score = 0
        else:
            self.turn_masks[0, self.ego_id] = 1

        return done

    def setup_data(self):
        self.turn_obs = np.zeros(
            (self.n_rollout_threads, *self.buffer.obs.shape[2:]), dtype=np.float32
        )
        self.turn_share_obs = np.zeros(
            (self.n_rollout_threads, *self.buffer.share_obs.shape[2:]), dtype=np.float32
        )
        self.turn_available_actions = np.zeros(
            (self.n_rollout_threads, *self.buffer.available_actions.shape[2:]),
            dtype=np.float32,
        )
        self.turn_values = np.zeros(
            (self.n_rollout_threads, *self.buffer.value_preds.shape[2:]),
            dtype=np.float32,
        )
        self.turn_actions = np.zeros(
            (self.n_rollout_threads, *self.buffer.actions.shape[2:]), dtype=np.float32
        )
        self.turn_action_log_probs = np.zeros(
            (self.n_rollout_threads, *self.buffer.action_log_probs.shape[2:]),
            dtype=np.float32,
        )
        self.turn_rnn_states = np.zeros(
            (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        self.turn_rnn_states_critic = np.zeros_like(self.turn_rnn_states)
        self.turn_masks = np.ones(
            (self.n_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32
        )
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros(
            (self.n_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32
        )

        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.env.reset()

        # replay buffer
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()
        self.running_score = 0

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        # self.buffer.chooseafter_update()
        self.buffer.reset_after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        print("SAVED TO", self.save_dir)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor.pt")
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + "/critic.pt")
            self.policy.critic.load_state_dict(policy_critic_state_dict)
