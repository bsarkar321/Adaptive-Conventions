from overcooked_env.overcooked_env import SimplifiedOvercooked

from garage.envs import GymEnv

from garage.experiment import TaskSampler
from garage.sampler.env_update import SetTaskUpdate

import numpy as np

LAYOUT = 'random1'

PARTNERS = []


class GarageOvercooked(GymEnv):

    def __init__(self, partners=None, state=None, partner_id=None):
        env = SimplifiedOvercooked(
            LAYOUT,
            ego_agent_idx=1,
            baselines=False,
            use_rew_shape=True
        )

        env.partners = [[None]]

        # self.partners = (partners if partners is not None else [])
        if partners is not None:
            global PARTNERS
            PARTNERS = partners

        self.partner_id = partner_id

        if state is not None:
            env.base_env.state = state

        super().__init__(env, max_episode_length=200)

        if partner_id is not None:
            self.choose_partner(self.partner_id)

    def choose_partner(self, agent_id):
        partner = PARTNERS[agent_id]
        self.partner_id = agent_id
        self._env.partners[0][0] = partner

    def set_task(self, id):
        self.choose_partner(id)
        self._env.task_name = str(id)

    def add_partner_agent(self, agent):
        PARTNERS.append(agent)

    def get_all_partners(self):
        return PARTNERS

    def __getstate__(self):
        return {
            '_state': self._env.base_env.state,
            '_partner_id': self.partner_id,
            '_partners': PARTNERS
        }

    def __setstate__(self, state):
        self.__init__(
            state['_partners'],
            state['_state'],
            state['_partner_id']
        )


class OvercookedTaskSampler(TaskSampler):

    def __init__(self, partner_indices, partner_copy):
        self.partner_indices = partner_indices
        self.partner_copy = partner_copy
        # self.env_copy = env

    def sample(self, n_tasks, with_replacement=True):
        if n_tasks == len(self.partner_indices):
            samples = self.partner_indices
        else:
            samples = np.random.choice(
                self.partner_indices,
                size=n_tasks,
                replace=with_replacement
            )
        return [
            SetTaskUpdate(GarageOvercooked, x, None) for x in samples
        ]

    @property
    def n_tasks(self):
        return len(self.partner_indices)

class GarageOvercookedCombined(GymEnv):

    def __init__(self, partners=None, state=None, partner_id=None):
        env = SimplifiedOvercooked(
            LAYOUT,
            ego_agent_idx=1,
            baselines=False,
            use_rew_shape=True
        )

        env.partners = [[]]

        # self.partners = (partners if partners is not None else [])
        if partners is not None:
            env.partners[0] = partners

        self.partner_id = partner_id

        if state is not None:
            env.base_env.state = state

        super().__init__(env, max_episode_length=200)

        if partner_id is not None:
            self.choose_partner(self.partner_id)

    def add_partner_agent(self, agent):
        self._env.add_partner_agent(agent, 0)

    def get_all_partners(self):
        return self._env.partners[0]

    def __getstate__(self):
        return {
            '_state': self._env.base_env.state,
            '_partner_id': self.partner_id,
            '_partners': self._env.partners[0]
        }

    def __setstate__(self, state):
        self.__init__(
            state['_partners'],
            state['_state'],
            state['_partner_id']
        )
