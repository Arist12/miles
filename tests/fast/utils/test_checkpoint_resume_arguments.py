from argparse import Namespace

import pytest

from miles.utils.arguments import _resolve_checkpoint_resume


def _args(tmp_path, **overrides):
    values = {
        "load": None,
        "ref_load": None,
        "hf_checkpoint": str(tmp_path / "base"),
        "start_rollout_id": None,
        "lora_adapter_path": None,
        "rollout_global_dataset": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_fresh_bridge_run_starts_at_zero(tmp_path):
    args = _args(tmp_path)
    _resolve_checkpoint_resume(args)
    assert args.load == args.hf_checkpoint
    assert args.start_rollout_id == 0
    assert args.lora_resume_root is None


def test_lora_checkpoint_resumes_its_run(tmp_path):
    adapter = tmp_path / "run" / "iter_0000007" / "adapter"
    args = _args(tmp_path, lora_adapter_path=str(adapter))

    _resolve_checkpoint_resume(args)

    assert args.lora_resume_root == str(tmp_path / "run")
    assert args.start_rollout_id is None


@pytest.mark.parametrize(
    "adapter, overrides",
    [
        ("released-adapter", {}),
        ("run/iter_7/adapter", {}),
        ("run/iter_0000007/adapter", {"rollout_global_dataset": False}),
        ("run/iter_0000007/adapter", {"finetune": True}),
    ],
    ids=["not-a-checkpoint", "bad-iteration-dir", "custom-data-source", "finetune"],
)
def test_other_adapters_are_weight_only_warm_starts(tmp_path, adapter, overrides):
    args = _args(tmp_path, lora_adapter_path=str(tmp_path / adapter), **overrides)

    _resolve_checkpoint_resume(args)

    assert args.lora_resume_root is None
    assert args.start_rollout_id == 0


def test_bridge_training_checkpoint_preserves_automatic_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("7")
    args = _args(tmp_path, load=str(checkpoint))

    _resolve_checkpoint_resume(args)

    assert args.load == str(checkpoint)
    assert args.start_rollout_id is None
