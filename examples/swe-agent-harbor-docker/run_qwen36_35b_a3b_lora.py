"""Qwen3.6-35B-A3B LoRA agentic RL on one node, Harbor docker sandboxes (validated on 8x MI355X).

Synchronous colocated GRPO: the policy serves through the Miles session server (TITO,
``--tito-model qwen36``), a Harbor agent server on the same host runs each trajectory in
its own docker sandbox and returns the verifier's reward. Loss defaults follow the Mercor
APEX-Agents recipe: DPPO (binary TV, delta 0.15) against the rollout log-probs,
prompt_mean aggregation, no KL, no std normalisation; LoRA lr is 10x their full-parameter
1e-6 (LoRA Without Regret).

Start the agent server first (see README.md), then inside the Miles image:

    python examples/swe-agent-harbor-docker/run_qwen36_35b_a3b_lora.py \
        --hf-checkpoint /models/Qwen3.6-35B-A3B --prompt-data /data/tasks.jsonl \
        --agent-server-url http://127.0.0.1:11000

Training records are ``{"prompt": ..., "metadata": {"instance_id": <task dir>, "agent_name": "terminus-2"}}``.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.external_utils import command_utils

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    megatron_model_type: str = "qwen3.6-35B-A3B_lora"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    hf_checkpoint: str = "/root/models/Qwen3.6-35B-A3B"
    prompt_data: str = "/root/agentic_train.jsonl"
    save_dir: str = ""
    save_interval: int = 20
    save_traces_dir: str = ""

    # rollout shape: rollout_batch_size prompts x n_samples_per_prompt trajectories per step
    num_rollout: int = 100
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    max_seq_len: int = 32768
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lr: float = 1e-5

    # loss
    use_dppo: bool = True
    loss_aggregation: str = "prompt_mean"
    # Groups with an infrastructure failure are always dropped. With over-sampling, also
    # keep only groups whose rewards differ and sample more prompts until the batch is
    # full: Mercor's zero-variance filter with sample_full_batch. 0 disables the latter.
    over_sampling_batch_size: int = 8

    # rollout engine
    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.5
    session_server_workers: int = 32

    # agent server
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", "http://127.0.0.1:11000")
    agent_trial_timeout: int = 3600

    use_wandb: bool = True
    wandb_project: str = "miles-agentic-qwen36"
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    extra_args: str = ""


def execute(args: ScriptArgs):
    U = args.create_backend()

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
    if args.save_dir:
        ckpt_args += f"--save {args.save_dir}/{args.run_id} --save-interval {args.save_interval} "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        '--target-modules "all-linear" --experts-shared-outer-loras --lora-base-cpu-backup '
        "--no-gradient-accumulation-fusion "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} --input-key prompt --metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--rollout-temperature {args.rollout_temperature} --rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
    )

    # GRPO without std normalisation or KL, as in the Mercor recipe
    algo_args = "--advantage-estimator grpo --disable-grpo-std-normalization --entropy-coef 0.0 "
    algo_args += "--use-rollout-logprobs "
    if args.use_dppo:
        algo_args += "--use-dppo --dppo-delta-low 0.15 --dppo-delta-high 0.15 "
    else:
        algo_args += "--eps-clip 0.2 --eps-clip-high 0.28 "
    algo_args += f"--loss-aggregation {args.loss_aggregation} "

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.01 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0 "
    )

    # TP <= 2 (two query groups); megatron-core GatedDeltaNet needs unpacked bshd batches
    perf_args = (
        "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--qkv-format bshd --micro-batch-size 1 --max-tokens-per-gpu {args.max_seq_len} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-context-length {args.max_seq_len} "
        "--sglang-dtype bfloat16 --sglang-decode-log-interval 1000 "
        f"--sglang-max-lora-rank {args.lora_rank} --sglang-lora-backend triton "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
        "--sglang-router-port 31000 "
    )
    if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm" or Path("/opt/rocm").exists():
        # ROCm shared-expert fusion needs per-expert LoRA factors; shared-outer has none.
        sglang_args += "--sglang-disable-shared-experts-fusion "

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--use-session-server --tito-model qwen36 "
        f"--session-server-port 30000 --session-server-workers {args.session_server_workers} "
    )
    if args.over_sampling_batch_size:
        agent_args += (
            "--dynamic-sampling-filter-path generate.filter_infra_failures_and_zero_std "
            f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        )
    else:
        agent_args += "--dynamic-sampling-filter-path generate.filter_infra_failures "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
        "--observe-training-entropy --log-passrate "
    )
    if args.save_traces_dir:
        misc_args += f"--dump-details {args.save_traces_dir}/{args.run_id} "

    wandb_args = ""
    if args.use_wandb:
        # Without WANDB_API_KEY the SDK falls back to ~/.netrc, which keeps the key off
        # the logged command line.
        wandb_args = (
            f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.run_id} "
            "--disable-wandb-random-suffix "
        )
        if os.environ.get("WANDB_API_KEY"):
            wandb_args += f"--wandb-key {os.environ['WANDB_API_KEY']} "
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    train_args = (
        f"{ckpt_args}{lora_args}{rollout_args}{algo_args}{optimizer_args}{perf_args}"
        f"{sglang_args}{agent_args}{misc_args}{wandb_args}{args.extra_args}"
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
            "AGENT_SERVER_URL": args.agent_server_url,
            "AGENT_MODEL_NAME": "model",
            "AGENT_TRIAL_TIMEOUT": str(args.agent_trial_timeout),
            # Ray empties HIP/CUDA_VISIBLE_DEVICES in GPU-less actors such as the worker
            # manager, which imports every worker class; on ROCm `import aiter` (pulled in
            # by sglang under SGLANG_USE_AITER=1) fails without a visible device. The GPU
            # actors already run with these set and pick their devices themselves.
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
