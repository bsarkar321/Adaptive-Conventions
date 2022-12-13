from garage.torch.algos.maml_ppo import MAMLPPO

import torch
from garage.torch._functions import zero_optim_grads
from garage.torch import update_module_params

from MAPPO.utils.util import update_linear_schedule

class CustomMAML(MAMLPPO):
    def __init__(self,
                 env,
                 policy,
                 value_function,
                 sampler,
                 task_sampler,
                 inner_lr=1e-1,
                 outer_lr=1e-3,
                 lr_clip_range=5e-1,
                 discount=0.99,
                 gae_lambda=1.0,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 meta_batch_size=20,
                 num_grad_updates=1,
                 meta_evaluator=None,
                 evaluate_every_n_epochs=1,
                 max_optimization_epochs=1,
                 total_iters=100):
        super().__init__(
            env,
            policy,
            value_function,
            sampler,
            task_sampler,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            lr_clip_range=lr_clip_range,
            discount=discount,
            gae_lambda=gae_lambda,
            center_adv=center_adv,
            positive_adv=positive_adv,
            policy_ent_coeff=policy_ent_coeff,
            use_softplus_entropy=use_softplus_entropy,
            stop_entropy_gradient=stop_entropy_gradient,
            entropy_method=entropy_method,
            meta_batch_size=meta_batch_size,
            num_grad_updates=num_grad_updates,
            meta_evaluator=meta_evaluator,
            evaluate_every_n_epochs=evaluate_every_n_epochs
        )

        self.init_outer_lr = outer_lr

        self.max_optimization_epochs = max_optimization_epochs
        self.total_iters = total_iters
        self.cur_episode = 0

    def _train_once(self, trainer, all_samples, all_params):
        """Train the algorithm once.
        Args:
            trainer (Trainer): The experiment runner.
            all_samples (list[list[_MAMLEpisodeBatch]]): A two
                dimensional list of _MAMLEpisodeBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
        Returns:
            float: Average return.
        """
        itr = trainer.step_itr
        old_theta = dict(self._policy.named_parameters())

        kl_before = self._compute_kl_constraint(all_samples,
                                                all_params,
                                                set_grad=False)

        meta_objective = self._compute_meta_loss(all_samples, all_params)

        for i in range(self.max_optimization_epochs):
            if i == 0:
                meta_objective_i = meta_objective
            else:
                meta_objective_i = self._compute_meta_loss(all_samples, all_params)

            zero_optim_grads(self._meta_optimizer)
            meta_objective_i.backward()
            # self._meta_optimize(all_samples, all_params)
            self._meta_optimizer.step()
        update_linear_schedule(self._meta_optimizer, self.cur_episode, self.total_iters, self.init_outer_lr)
        self.cur_episode += 1

        # Log
        loss_after = self._compute_meta_loss(all_samples,
                                             all_params,
                                             set_grad=False)
        kl_after = self._compute_kl_constraint(all_samples,
                                               all_params,
                                               set_grad=False)

        with torch.no_grad():
            policy_entropy = self._compute_policy_entropy(
                [task_samples[0] for task_samples in all_samples])
            average_return = self._log_performance(
                itr, all_samples, meta_objective.item(), loss_after.item(),
                kl_before.item(), kl_after.item(),
                policy_entropy.mean().item())

        if self._meta_evaluator and itr % self._evaluate_every_n_epochs == 0:
            self._meta_evaluator.evaluate(self)

        update_module_params(self._old_policy, old_theta)

        return average_return
