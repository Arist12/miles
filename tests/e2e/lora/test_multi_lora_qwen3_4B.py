"""Eight-GPU multi-LoRA E2E with two independently stepped adapters.

Exercises the current disaggregated Bridge path end to end: concurrent rollout,
per-slot optimization, a selective post-step SGLang upsert, automatic retirement,
and both native and HF checkpoint export.
"""

import os
import shutil
from pathlib import Path

import torch

from examples.multi_lora import run_multi_lora
from safetensors.torch import load_file
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=500, suite="stage-c-8-gpu-h100", labels=["lora", "fully-async"])

MODEL_DIR = Path("/root/models/Qwen3-4B")
CONFIG_DIR = Path("/tmp/multi_lora_ci")
SAVE_DIR = Path("/root/checkpoints/multi-lora-qwen3-4B-ci")

_ADAPTERS = {
    "gsm8k": {
        "data": "/root/datasets/gsm8k/train.parquet",
        "input_key": "messages",
        "label_key": "label",
        "rm_type": "math",
        "num_step": 3,
    },
    "dapo_math": {
        "data": "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
        "input_key": "prompt",
        "label_key": "label",
        "rm_type": "deepscaler",
        "num_step": 2,
    },
}


def prepare() -> None:
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/Qwen3-4B --local-dir {MODEL_DIR}")
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir="/root/datasets")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir="/root/datasets")


def _write_adapter_configs() -> None:
    shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    CONFIG_DIR.mkdir(parents=True)
    for name, dataset in _ADAPTERS.items():
        save_dir = SAVE_DIR / name
        (CONFIG_DIR / f"{name}.yaml").write_text(
            "\n".join(
                [
                    "rank: 8",
                    "alpha: 8",
                    "rollout_batch_size: 2",
                    "n_samples_per_prompt: 4",
                    f"data: {dataset['data']}",
                    f"input_key: {dataset['input_key']}",
                    f"label_key: {dataset['label_key']}",
                    f"rm_type: {dataset['rm_type']}",
                    f"num_step: {dataset['num_step']}",
                    f"save: {save_dir}",
                    "",
                ]
            )
        )


def execute() -> None:
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    _write_adapter_configs()
    run_multi_lora._ADAPTER_DIR = str(CONFIG_DIR)
    args = run_multi_lora.ScriptArgs(
        hf_checkpoint=str(MODEL_DIR),
        save_dir=str(SAVE_DIR),
        n_adapters=2,
        adapters="gsm8k,dapo_math",
        num_rollout=10,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        rollout_max_response_len=1024,
        global_batch_size=16,
        save_interval=1,
        extra_args="--sglang-context-length 1536",
    )
    run_multi_lora._train(args, service=False)

    adapter_b_states = {}
    for name, config in _ADAPTERS.items():
        checkpoint = SAVE_DIR / name / "checkpoints" / f"step_{config['num_step']}"
        assert (checkpoint / "adapter_config.json").is_file()
        assert {path.name for path in checkpoint.glob("adapter_megatron_*.pt")} == {
            "adapter_megatron_tp0_pp0.pt",
            "adapter_megatron_tp1_pp0.pt",
        }
        peft_state = load_file(checkpoint / "adapter_model.safetensors")
        assert peft_state and all(torch.isfinite(value).all() for value in peft_state.values())
        assert any("lora_A" in key for key in peft_state)
        adapter_b_states[name] = {key: value for key, value in peft_state.items() if "lora_B" in key}
        assert adapter_b_states[name]

    common_keys = set.intersection(*(set(state) for state in adapter_b_states.values()))
    assert common_keys
    assert any(torch.count_nonzero(value) > 0 for value in adapter_b_states["gsm8k"].values())
    assert any(
        not torch.equal(adapter_b_states["gsm8k"][key], adapter_b_states["dapo_math"][key]) for key in common_keys
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    try:
        execute()
    finally:
        U.exec_command_cpu("ray stop --force || true; pkill -9 sglang || true")
