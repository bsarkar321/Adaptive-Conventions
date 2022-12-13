import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from pantheonrl.common.multiagentenv import SimultaneousEnv


def convert_action(action):
    # return action[0]
    if not isinstance(action, np.ndarray):
        return action
    return action[0]


class PantheonOvercooked(SimultaneousEnv):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False, use_rew_shape=True):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the
        'both_agent_obs' field
        """
        super().__init__()

        DEFAULT_ENV_PARAMS = {"horizon": 200}
        if use_rew_shape:
            rew_shaping_params = {
                "PLACEMENT_IN_POT_REW": 3,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "DISH_DISP_DISTANCE_REW": 0,
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
            }
        else:
            rew_shaping_params = {
                "PLACEMENT_IN_POT_REW": 0,
                "DISH_PICKUP_REWARD": 0,
                "SOUP_PICKUP_REWARD": 0,
                "DISH_DISP_DISTANCE_REW": 0,
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
            }

        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name=layout_name, rew_shaping_params=rew_shaping_params
        )
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            self.mdp, NO_COUNTERS_PARAMS, force_compute=False
        )

        self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        # self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)

        if baselines:
            np.random.seed(0)

        self.lA = len(Action.ALL_ACTIONS)
        self.observation_space = self._setup_observation_space()
        self.share_observation_space = self.observation_space
        self.action_space = gym.spaces.Discrete(self.lA)
        self.ego_ind = ego_agent_idx
        self.multi_reset()

    def get_mask(self):
        return np.array([1] * self.lA, dtype=bool)

    def featurize_fn(self, x):
        ob1, ob2 = self.mdp.featurize_state(x, self.mlp)
        return (ob1, ob1, self.get_mask()), (ob2, ob2, self.get_mask())

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0][0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf
        print(high.shape)

        return gym.spaces.Box(-high, high, dtype=np.float32)

    def get_full_obs(self):
        pass

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in
            index format encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        # print(f"{ego_action}, {type(ego_action)}, {alt_action}, {type(alt_action)}")

        
        ego_action = convert_action(ego_action)
        alt_action = convert_action(alt_action)

        joint_action = (
            Action.INDEX_TO_ACTION[ego_action],
            Action.INDEX_TO_ACTION[alt_action],
        )

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info["shaped_r"]
        reward = reward + rew_shape

        # print(self.base_env.mdp.state_string(next_state))
        return self.featurize_fn(next_state), (reward, reward), done, {}  # info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which
        agent is assigned to which starting location, in order to make
        sure that the agents are trained to be able to complete the
        task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize
        starting positions, and not have to deal with randomizing
        indices.
        """
        self.base_env.reset()
        return self.featurize_fn(self.base_env.state)

    def render(self, mode="human", close=False):
        pass


class DecentralizedOvercooked(PantheonOvercooked):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False, use_rew_shape=True):
        super().__init__(layout_name, ego_agent_idx, baselines, use_rew_shape)

    def featurize_fn(self, x):
        my_obs, ot_obs = super().featurize_fn(x)
        return (my_obs[0], my_obs[2]), (ot_obs[0], ot_obs[2])


class SimplifiedOvercooked(PantheonOvercooked):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False, use_rew_shape=True):
        super().__init__(layout_name, ego_agent_idx, baselines, use_rew_shape)
        self.task_name = "0"

    def multi_step(self, ego_action, alt_action):
        ((o0, _, _), (o1, _, _)), r, d, i = super().multi_step(ego_action, alt_action)
        return (o0, o1), r, d, {"task_name": self.task_name}

    def multi_reset(self):
        ((o0, _, _), (o1, _, _)) = super().multi_reset()
        return (o0, o1)
