"""
SWE-Agent: reward, metrics, and rollout class.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run

Task-type agnostic — reward is pre-computed by the Harbor environment
and stored in sample.metadata["reward"] regardless of task type.

Dynamic filter uses the general-purpose ``check_no_aborted`` from
``miles.rollout.filter_hub.dynamic_sampling_filters``.

Components:
  - reward_func: reads pre-computed reward from sample metadata
  - aggregate_agent_metrics: aggregates agent timing/count metrics
  - log_rollout_agent_metrics: --custom-rollout-log-function-path hook that logs them
    in both synchronous and fully-async mode
  - RolloutFn: InferenceRolloutFn subclass that logs agent metrics
"""

import logging

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Infrastructure failures --


# Unscored trials the policy is answerable for: it ran out of context or out of its time
# budget. These keep their 0.0 reward, as Terminal-Bench scores them.
_POLICY_LIMIT_EXIT_STATUSES = ("SequenceLengthLimitExceeded", "TimeLimitExceeded")


def is_infra_failure(sample: Sample) -> bool:
    """The verifier never scored the trial, for a reason other than the policy exhausting
    its context or time budget: an environment that failed to build or start, a sandbox or
    harness crash, or an unreachable agent server. Its 0.0 reward says nothing about the
    policy."""
    metadata = sample.metadata or {}
    if metadata.get("eval_report"):
        return False
    return metadata.get("exit_status") not in _POLICY_LIMIT_EXIT_STATUSES


def filter_infra_failures(args, samples: list[Sample | list[Sample]], **kwargs) -> FilterOutput:
    """Drop a group in which any trial failed for infrastructure reasons."""
    if any(is_infra_failure(sample) for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="infra_failure")
    return FilterOutput(keep=True)


def filter_infra_failures_and_zero_std(args, samples: list[Sample | list[Sample]], **kwargs) -> FilterOutput:
    """``filter_infra_failures``, then keep only groups whose rewards differ."""
    output = filter_infra_failures(args, samples, **kwargs)
    if not output.keep:
        return output
    return apply_reward_nonzero_std_filter(args, samples, **kwargs)


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the agent environment during generate().

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Agent Metrics Aggregation --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    """Values agents reported for ``key``; agents that omit it are skipped.

    Defaulting to 0 would log a fake measurement and drag the batch mean down.
    Callers skip empty lists, so an unreported key stays out of the log.
    """
    return [value for metrics in all_metrics if (value := metrics.get(key)) is not None]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def _with_derived_timings(metrics: dict) -> dict:
    """Fill the model/environment split from per-request latencies when the agent did not
    report it. terminus-2 reports ``api_request_times_msec`` (one entry per model call) and
    Harbor the phase durations; time in the agent phase outside model calls is tool and
    sandbox execution plus harness overhead."""
    out = dict(metrics)
    requests = out.get("api_request_times_msec")
    if isinstance(requests, list) and requests:
        out.setdefault("model_query_time_sum", sum(requests) / 1000)
        out.setdefault("model_query_time_avg", sum(requests) / 1000 / len(requests))
    run, model = out.get("agent_run_time"), out.get("model_query_time_sum")
    if run and model is not None:
        out.setdefault("env_execution_time_sum", max(run - model, 0.0))
        out.setdefault("model_time_ratio", min(model / run, 1.0))
        out.setdefault("env_time_ratio", max(1.0 - model / run, 0.0))
    if run and out.get("turns"):
        out.setdefault("time_per_turn", run / out["turns"])
    return out


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        _with_derived_timings(s.metadata["agent_metrics"])
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(
        metrics,
        all_metrics,
        ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time", "env_setup_time"],
    )
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    statuses = [s.metadata.get("exit_status", "") for s in samples if s.metadata]
    for status in sorted(set(statuses)):
        metrics[f"agent/exit_status/{status or 'none'}"] = statuses.count(status) / len(statuses)

    return metrics


def log_rollout_agent_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """``--custom-rollout-log-function-path`` hook: add the agent metrics to the rollout log.

    Runs for every trained rollout in synchronous and fully-async mode alike; fully async
    installs its own rollout class, so ``RolloutFn`` never sees those batches.
    """
    flat = [s for item in samples for s in (item if isinstance(item, list) else [item])]
    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(aggregate_agent_metrics(flat))
    return False  # keep miles' default rollout logging, which includes these metrics


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    """Rollout function with agent metrics aggregation."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        agent_metrics = aggregate_agent_metrics(all_samples)
        if agent_metrics:
            metrics = output.metrics or {}
            metrics.update(agent_metrics)
            output.metrics = metrics
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")

        return output
