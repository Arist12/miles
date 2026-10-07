"""CPU tests for the DPPO policy loss and ``--loss-aggregation prompt_mean``."""

from __future__ import annotations

import math

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.math_utils import compute_dppo_policy_loss
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data


@pytest.fixture(autouse=True)
def _parallel_state():
    make_parallel_state()


def _reference_binary_tv(log_probs, mu_log_probs, advantages, delta_low, delta_high):
    """SkyRL's dppo_policy_loss (binary_tv), the implementation the Mercor recipe trained with."""
    ratio = torch.exp(log_probs - mu_log_probs)
    prob_diff = torch.exp(log_probs) - torch.exp(mu_log_probs)
    mask = torch.ones_like(advantages)
    mask[(advantages > 0) & (prob_diff > delta_high)] = 0.0
    mask[(advantages < 0) & (-prob_diff > delta_low)] = 0.0
    return -(ratio * advantages * mask), (mask == 0).float()


class TestDppoPolicyLoss:
    def test_binary_tv_matches_reference(self):
        torch.manual_seed(0)
        n = 4096
        mu = torch.log(torch.rand(n).clamp_min(1e-4))
        log_probs = torch.log((mu.exp() + 0.3 * torch.randn(n)).clamp(1e-4, 1.0))
        advantages = torch.randn(n)
        got, got_masked = compute_dppo_policy_loss(log_probs, mu, advantages, 0.15, 0.15, "binary_tv")
        want, want_masked = _reference_binary_tv(log_probs, mu, advantages, 0.15, 0.15)
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(got_masked, want_masked)
        assert 0 < got_masked.mean().item() < 1, "the fixture should exercise both branches"

    def test_mask_only_in_advantage_direction(self):
        mu = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        # p moved up by 0.3 on tokens 0/1 and down by 0.3 on tokens 2/3
        log_probs = torch.log(torch.tensor([0.8, 0.8, 0.2, 0.2]))
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
        _, masked = compute_dppo_policy_loss(log_probs, mu, advantages, 0.15, 0.15)
        # up-move masked only where A > 0, down-move only where A < 0
        assert masked.tolist() == [1.0, 0.0, 0.0, 1.0]

    def test_masked_tokens_carry_no_gradient_and_ratio_is_unclipped(self):
        mu = torch.log(torch.tensor([0.5, 0.1]))
        log_probs = torch.log(torch.tensor([0.9, 0.2])).requires_grad_(True)
        advantages = torch.tensor([1.0, 1.0])
        loss, masked = compute_dppo_policy_loss(log_probs, mu, advantages, 0.15, 0.15)
        loss.sum().backward()
        assert masked.tolist() == [1.0, 0.0]
        assert log_probs.grad[0].item() == 0.0
        # d(-ratio * A)/d logp = -ratio * A, with ratio = 2.0 (outside PPO's 1.28 clip)
        assert math.isclose(log_probs.grad[1].item(), -2.0, rel_tol=1e-5)

    def test_binary_kl_requires_divergence_in_advantage_direction(self):
        mu = torch.log(torch.tensor([0.5, 0.5]))
        log_probs = torch.log(torch.tensor([0.95, 0.05]))
        advantages = torch.tensor([1.0, 1.0])
        _, masked = compute_dppo_policy_loss(log_probs, mu, advantages, 0.15, 0.15, "binary_kl")
        assert masked.tolist() == [1.0, 0.0]

    def test_extreme_log_ratios_stay_finite(self):
        mu = torch.tensor([0.0, -50.0, 0.0])
        log_probs = torch.tensor([-50.0, 0.0, float("nan")])
        loss, masked = compute_dppo_policy_loss(log_probs, mu, torch.ones(3), 0.15, 0.15)
        assert torch.isfinite(loss).all()
        assert torch.isfinite(masked).all()


def _convert(samples, **arg_overrides):
    return convert_samples_to_train_data(
        make_args(rewards_normalization=False, **arg_overrides),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


class TestPromptMean:
    def _samples(self):
        # prompt 0: two rollouts with 2 and 6 loss tokens; prompt 1: two rollouts with 3 and 1
        lens = [(0, 2), (0, 6), (1, 3), (1, 1)]
        return [make_sample(group_index=g, index=i, response_length=n) for i, (g, n) in enumerate(lens)]

    def test_default_stays_per_rollout(self):
        out = _convert(self._samples())
        assert out["rollout_mask_sums"] == [2, 6, 3, 1]

    def test_denominators(self):
        out = _convert(self._samples(), loss_aggregation="prompt_mean")
        # P = 2 groups over R = 4 rollouts -> scale 1/2; T_0 = 8, T_1 = 4
        assert out["rollout_mask_sums"] == [4.0, 4.0, 2.0, 2.0]

    def test_reducer_gives_mean_over_prompts_of_token_means(self):
        samples = self._samples()
        out = _convert(samples, loss_aggregation="prompt_mean")
        lens = out["response_lengths"]
        masks = [torch.tensor(m, dtype=torch.int) for m in out["loss_masks"]]
        x = torch.arange(1.0, sum(lens) + 1)  # distinct per-token values
        reducer = get_sum_of_sample_mean(
            lens, lens, masks, denominators=torch.tensor(out["rollout_mask_sums"], dtype=torch.float32)
        )
        num_rollouts = len(samples)
        got = reducer(x).item() / num_rollouts  # objective.loss_function divides by the rollout count
        per_sample = x.split(lens)
        prompt0 = torch.cat(per_sample[:2]).mean()
        prompt1 = torch.cat(per_sample[2:]).mean()
        assert math.isclose(got, ((prompt0 + prompt1) / 2).item(), rel_tol=1e-6)

    def test_masked_tokens_excluded_from_group_total(self):
        samples = self._samples()
        samples[1].loss_mask = [1, 1, 0, 0, 0, 0]
        out = _convert(samples, loss_aggregation="prompt_mean")
        assert out["rollout_mask_sums"][:2] == [2.0, 2.0]
