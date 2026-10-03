"""GLM-5.2 744B-A40B GRPO LoRA RL for AMD (MI350X / MI355X), multi-node.

ROCm counterpart of scripts/run_glm5_2_744b_a40b_lora.py: TP intra-node, EP across the
whole world. The trainer runs megatron-core's unfused DSA, as the published CUDA run did;
SGLang serves it with its TileLang DSA kernels (flashmla has no HIP build).
GLM-5.2_5layer is a mechanics smoke only (zero reward).

Examples:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py prepare --model-name GLM-5.2
  # 2 nodes, Ray already up across them, MILES_SCRIPT_EXTERNAL_RAY=1:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py train --model-name GLM-5.2 --num-nodes 2
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = {
    "GLM-5.2": "zai-org/GLM-5.2",
    "GLM-5.2_5layer": "Pinaster/GLM-5.2_5layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.2": "glm5.2-744B-A40B_lora",
    "GLM-5.2_5layer": "glm5.2-744B-A40B_5layer_lora",
}

_NUM_LAYERS = {"GLM-5.2": 78, "GLM-5.2_5layer": 5}

# (seq_length, rollout_max_response_len), as in the CUDA recipe.
_TASK_SEQ = {"dapo-math": (8192, 4096), "gsm8k": (1024, 512)}

# Micro-batches pad to tp_size * this; miles' 128 trains a ~360-token gsm8k sample as 1024.
_TASK_PAD_MULTIPLIER = {"dapo-math": 0, "gsm8k": 16}  # 0 => miles' default

# The TileLang DSA decode kernel needs >= 4 of GLM-5.2's 64 heads per engine rank.
_MAX_ENGINE_GPUS = 16


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["GLM-5.2", "GLM-5.2_5layer"] = "GLM-5.2"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None
    task: Literal["dapo-math", "gsm8k"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    dsa_attention_backend: Literal["megatron", "tilelang"] = "megatron"
    # R3 rollout routing replay (arxiv 2510.11370)
    use_r3: bool = True

    # The published CUDA run's rank. Attention + MLP/expert gate/up; the DSA indexer is not an
    # HF attention target, and down_proj drifts the train/rollout logprob gap.
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "attn,mlp"
    exclude_modules: str = "down_proj"
    # MLP gate/up adapters on the last N layers only, as in the published CUDA run; 0 => all.
    # On every layer the train/rollout logprob gap grows with the adapters (0.014 -> 0.031
    # over 40 gsm8k rollouts at 4 nodes).
    mlp_lora_last_n: int = 10
    lora_base_cpu_backup: bool = True
    experts_shared_outer_loras: bool = True

    num_rollout: int = 40
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    # 0 => the task default above
    seq_length: int = 0
    rollout_max_response_len: int = 0
    # -1 => the task default above; 0 => miles' default
    data_pad_size_multiplier: int = -1
    lr: float = 1e-5

    # 0 => min(world, _MAX_ENGINE_GPUS)
    rollout_num_gpus_per_engine: int = 0
    sglang_mem_fraction_static: float = 0.85
    # 0 => seq_length; otherwise SGLang sizes req_to_token for the native 1 Mi context.
    sglang_context_length: int = 0
    sglang_max_running_requests: int = 0  # 0 => rollout_batch_size * n_samples_per_prompt

    # Loading the 744B checkpoint outlasts miles' 10-minute default.
    distributed_timeout_minutes: int = 60

    # The engine's first base-weight release to pinned host memory takes minutes per rank.
    rollout_cell_tick_timeout: float = 3600.0
    rollout_cell_init_timeout: float = 5400.0

    offload_train: bool = True
    save_interval: int = 10
    save_dir: str = "/root/shared_data"
    # Node-local, not tmpfs: the engine already mirrors the base weights in host RAM.
    offload_train_disk_dir: str = ""  # "" => {save_dir}/train_offload
    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        seq, resp = _TASK_SEQ[self.task]
        if self.seq_length == 0:
            self.seq_length = seq
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = resp
        if self.data_pad_size_multiplier < 0:
            self.data_pad_size_multiplier = _TASK_PAD_MULTIPLIER[self.task]
        if not self.offload_train_disk_dir:
            self.offload_train_disk_dir = f"{self.save_dir}/train_offload"
        if self.sglang_context_length == 0:
            self.sglang_context_length = self.seq_length
        if self.sglang_max_running_requests == 0:
            self.sglang_max_running_requests = self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _set_rocm_environment() -> None:
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _resolve_num_gpus(args: ScriptArgs) -> tuple[str, int]:
    hardware = U.resolve_hardware(args)
    return hardware, args.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[hardware]


def _parallel_args(args: ScriptArgs, num_gpus: int) -> str:
    """TP intra-node, EP across the whole world (EP * ETP == TP * DP at PP = CP = 1)."""
    world_size = args.num_nodes * num_gpus
    qkv_format = "thd" if args.dsa_attention_backend == "tilelang" else "bshd"
    # Only thd carries the cross-layer DSA top-k that activation recompute needs.
    recompute = (
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        if qkv_format == "thd"
        else ""
    )
    return (
        f"--tensor-model-parallel-size {num_gpus} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {world_size} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 {recompute}"
    )


def _download_inputs(args: ScriptArgs) -> None:
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    backend.exec_command_cpu(f"hf download {_HF_REPO[args.model_name]} --local-dir {args.model_dir}/{args.model_name}")
    match args.task:
        case "dapo-math":
            backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
        case "gsm8k":
            backend.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _get_wandb_args(args: ScriptArgs) -> str:
    if not args.enable_wandb:
        return ""
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args and args.wandb_team:
        wandb_args += f"--wandb-team {shlex.quote(args.wandb_team)} "
    return wandb_args


def _execute(args: ScriptArgs) -> None:
    _set_rocm_environment()
    # execute_train only starts a local head; without this a multi-node job waits forever.
    if args.num_nodes > 1 and not os.environ.get("MILES_SCRIPT_EXTERNAL_RAY"):
        raise SystemExit(
            f"--num-nodes {args.num_nodes} needs a Ray cluster spanning all {args.num_nodes} nodes "
            "before this runs: start the head, join every worker with --num-gpus equal to the "
            "node's GPU count, wait for the full GPU count, then re-run with "
            "MILES_SCRIPT_EXTERNAL_RAY=1 and MASTER_ADDR set to the head's fabric IP."
        )
    hardware, num_gpus = _resolve_num_gpus(args)
    world_size = args.num_nodes * num_gpus
    engine_gpus = args.rollout_num_gpus_per_engine or min(world_size, _MAX_ENGINE_GPUS)
    print(
        f"[run] GLM-5.2 LoRA on {hardware}: model={args.model_name}, task={args.task}, "
        f"{args.num_nodes} node(s) x {num_gpus} GPUs (TP={num_gpus} EP={world_size}), "
        f"dsa={args.dsa_attention_backend}, seq={args.seq_length}, rollout tp={engine_gpus}"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )
    if args.save_interval > 0:
        ckpt_args += f"--save {args.save_dir}/{args.run_id} --save-interval {args.save_interval} "

    exclude_modules = [m for m in args.exclude_modules.split(",") if m]
    if args.mlp_lora_last_n > 0:
        first_adapted = max(0, _NUM_LAYERS[args.model_name] - args.mlp_lora_last_n)
        exclude_modules += [f"model.layers.{n}.mlp.*" for n in range(first_adapted)]
    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
        f"--target-modules {args.target_modules} --exclude-modules {shlex.quote(','.join(exclude_modules))} "
        "--no-gradient-accumulation-fusion "
    )
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    match args.task:
        case "dapo-math":  # {prompt, label} jsonl
            data_args = f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "
        case "gsm8k":  # {messages, label} parquet
            data_args = f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "

    rollout_args = (
        f"{data_args}--label-key label --apply-chat-template --rollout-shuffle --balance-data "
        "--rm-type math --rollout-temperature 1.0 "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--seq-length {args.seq_length} --rollout-max-context-len {args.seq_length} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )
    if args.data_pad_size_multiplier > 0:
        rollout_args += f"--data-pad-size-multiplier {args.data_pad_size_multiplier} "

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )

    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-ep-size {engine_gpus} "
        "--sglang-attention-backend dsa "
        # ROCm: flashmla_sparse / flashmla_kv have no HIP build.
        "--sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang "
        "--sglang-page-size 64 "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--sglang-cuda-graph-max-bs-decode {args.sglang_max_running_requests} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-chunked-prefill-size {min(8192, 2048 * engine_gpus)} --sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        f"--sglang-max-lora-rank {args.lora_rank} --sglang-lora-backend triton "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss "
        # RCCL's point-to-point path (AllToAll, isend/irecv) hangs on the dispatcher's exchange
        # across nodes.
        "--moe-token-dispatcher-type alltoall --moe-alltoall-via-allgather --colocate "
        f"--rollout-cell-tick-timeout {args.rollout_cell_tick_timeout} "
        f"--rollout-cell-init-timeout {args.rollout_cell_init_timeout} "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} "
        + (
            "--offload-train-target disk "
            f"--offload-train-disk-dir {args.offload_train_disk_dir} --offload-train-disk-chunk-mb 256 "
            if args.offload_train
            else "--no-offload-train "
        )
        +
        # bf16 has no grad scaler; let miles' own guard skip a non-finite step.
        "--no-check-for-nan-in-loss-and-grad "
        "--observe-training-entropy "
        f"--distributed-timeout-minutes {args.distributed_timeout_minutes} "
    )

    train_args = (
        f"{ckpt_args}{lora_args}{rollout_args}{optimizer_args}{grpo_args}{r3_args}"
        f"{_get_wandb_args(args)}{_parallel_args(args, num_gpus)}{sglang_args}{misc_args}{args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences.
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "SGLANG_NSA_FORCE_MLA": "1",
            # expandable_segments breaks torch_memory_saver under colocate.
            "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            # The ROCm image's NCCL_MIN_NCHANNELS=112 makes every communicator 64-112 channels
            # wide, and colocate rebuilds the trainer's communicators every rollout.
            "NCCL_MIN_NCHANNELS": "16",
            "NCCL_MAX_NCHANNELS": "16",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the model checkpoint and the task dataset. Run once per node."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Run GRPO LoRA training using prepared inputs."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download inputs and run GRPO LoRA training."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
