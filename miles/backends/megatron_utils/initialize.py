import logging
import random

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import _build_tokenizer, set_args

from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.hf_config import register_hf_config_aliases

from .ft.indep_dp import create_indep_dp_group
from .parallel import create_megatron_parallel_state

logger = logging.getLogger(__name__)


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * get_parallel_state().pp.rank)
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * get_parallel_state().effective_dp.rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.use_gloo_process_groups,
    )


def init(
    args,
    indep_dp_store_addr: str | None = None,
    indep_dp_info: IndepDPInfo | None = None,
):
    if indep_dp_info is None:
        indep_dp_info = IndepDPInfo.create_trivial()

    set_args(args)
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    # Pytorch distributed.
    _initialize_distributed(args)
    _warm_up_process_groups(args)
    if args.moe_ep_p2p_alltoall:
        from .ep_p2p_alltoall import install as install_ep_p2p_alltoall

        install_ep_p2p_alltoall(defer_transport=args.offload_train)

    indep_dp = create_indep_dp_group(
        store_addr=indep_dp_store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=dist.get_rank(),
        megatron_world_size=dist.get_world_size(),
    )

    set_parallel_state(create_megatron_parallel_state(indep_dp=indep_dp))

    # sanity check
    if getattr(args, "indep_dp", False):
        assert args.data_parallel_size == 1

    # Random seeds for reproducibility.
    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )
    register_hf_config_aliases()
    _build_tokenizer(args)
    # We won't use this. initialize to pass some validation in megatron.
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    if args.debug_deterministic_collective:
        assert not args.overlap_grad_reduce, "deterministic collectives require synchronous grad sync"

    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()

    if getattr(args, "custom_megatron_init_path", None):
        from miles.utils.function_registry import load_function

        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


# TODO shall we use a simpler method to determine which rank to init wandb?
def is_first_replica_megatron_main_rank():
    return (
        get_parallel_state().effective_dp_cp.rank == 0
        and get_parallel_state().tp.rank == 0
        and get_parallel_state().pp.rank == get_parallel_state().pp.size - 1
    )



_safe_ep_group = None
_safe_gloo_group = None
_retired_safe_ep_groups = []


def _warm_up_process_groups(args, *, install_hooks=True):
    """Eagerly connect process groups outside the pausable memory region."""
    from contextlib import nullcontext

    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    try:
        from torch_memory_saver import torch_memory_saver

        outside_region = torch_memory_saver.disable()
    except Exception as exc:  # no memory saver in this process: nothing to stay out of
        logger.info("[pg-warmup] torch_memory_saver unavailable (%s); warming in place", exc)
        outside_region = nullcontext()

    connected, skipped = [], []
    with outside_region:
        device = torch.cuda.current_device()
        probe = torch.ones(8, dtype=torch.float32, device=device)
        for name in sorted(n for n in dir(mpu) if n.startswith("get_") and n.endswith("_group")):
            try:
                group = getattr(mpu, name)()
            except Exception:
                # Several getters need arguments, and some assert when their feature is off.
                skipped.append(name)
                continue
            if not isinstance(group, dist.ProcessGroup):
                skipped.append(name)
                continue
            try:
                dist.all_reduce(probe.clone(), group=group)
                connected.append(name)
            except Exception as exc:  # a group we cannot warm is not a reason to fail the run
                logger.warning("[pg-warmup] %s did not warm: %s", name, exc)

        # Point-to-point setup for the group that actually stalls.
        try:
            ep_group = mpu.get_expert_model_parallel_group()
            ep_world = dist.get_world_size(group=ep_group)
            send = torch.zeros(ep_world * 8, dtype=torch.float32, device=device)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=ep_group)
            ep_note = f"ep_all_to_all_ok ranks={ep_world}"
        except Exception as exc:
            ep_note = f"ep_all_to_all_FAILED {exc}"

        torch.cuda.synchronize()
        dist.barrier()

    logger.info(
        "PG_WARMUP_DONE connected=%d skipped=%d %s outside_region=%s groups=%s",
        len(connected),
        len(skipped),
        ep_note,
        type(outside_region).__name__,
        ",".join(connected),
    )

    if install_hooks:
        _install_reload_warmup()
        _install_metadata_gather_gloo()
        _install_a2a_split_trace()
        _run_forward_outside_memory_region()


def _prepare_safe_ep_group(*, force=False):
    """Create fresh raw RCCL EP groups in globally consistent rank-set order."""
    from contextlib import nullcontext

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    global _safe_ep_group, _retired_safe_ep_groups
    if _safe_ep_group is not None and not force:
        return
    if _safe_gloo_group is None:
        raise RuntimeError("safe Gloo group must exist before the safe EP group")

    old_new_group = getattr(dist, "old_new_group", None)
    if old_new_group is None:
        raise RuntimeError("raw torch.distributed.new_group is unavailable")
    ep_group = mpu.get_expert_model_parallel_group()
    local_ep_ranks = tuple(dist.get_process_group_ranks(ep_group))
    memberships = [None] * dist.get_world_size()
    dist.all_gather_object(memberships, local_ep_ranks, group=_safe_gloo_group)
    all_ep_rank_sets = sorted({tuple(ranks) for ranks in memberships})
    try:
        from torch_memory_saver import torch_memory_saver

        outside_region = torch_memory_saver.disable()
    except Exception:
        outside_region = nullcontext()

    with outside_region:
        group = None
        global_rank = dist.get_rank()
        for ranks in all_ep_rank_sets:
            candidate = old_new_group(ranks=list(ranks))
            if global_rank in ranks:
                group = candidate
        if group is None:
            raise RuntimeError(
                f"rank {global_rank} is absent from EP memberships {all_ep_rank_sets}"
            )
        ep_world = len(local_ep_ranks)
        device = torch.cuda.current_device()
        send = torch.zeros(ep_world * 8, dtype=torch.float32, device=device)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        torch.cuda.synchronize()

    if _safe_ep_group is not None:
        # Do not let the old communicator destructor run while its device allocations
        # are in the post-pause state. There are only a few rollout iterations.
        _retired_safe_ep_groups.append(_safe_ep_group)
    _safe_ep_group = group
    logger.info(
        "SAFE_EP_GROUP_READY rank=%d group_rank=%d ranks=%s all_groups=%s",
        dist.get_rank(),
        dist.get_rank(group=group),
        local_ep_ranks,
        all_ep_rank_sets,
    )


def _install_metadata_gather_gloo():
    """Move the non-differentiable 256-element MoE count gather to Gloo."""
    import torch
    import torch.distributed as dist
    from megatron.core.tensor_parallel import mappings
    from miles.utils.distributed_utils import get_gloo_group

    global _safe_gloo_group
    _safe_gloo_group = get_gloo_group()

    if getattr(mappings.dist_all_gather_func, "_moe_metadata_gloo", False):
        return

    original = mappings.dist_all_gather_func

    def metadata_safe_all_gather(output, input_tensor, group=None, async_op=False):
        group_world = dist.get_world_size(group=group)
        if (
            isinstance(input_tensor, torch.Tensor)
            and input_tensor.dtype == torch.int64
            and input_tensor.numel() == 256
            and group_world == dist.get_world_size()
        ):
            if async_op:
                raise RuntimeError("MoE metadata Gloo gather requires async_op=False")
            cpu_input = input_tensor.detach().cpu()
            cpu_outputs = [torch.empty_like(cpu_input) for _ in range(group_world)]
            dist.all_gather(cpu_outputs, cpu_input, group=_safe_gloo_group)
            output.copy_(torch.cat(cpu_outputs, dim=0).to(input_tensor.device))
            logger.info(
                "MOE_METADATA_GLOO_GATHER rank=%d count=256 world=%d",
                dist.get_rank(),
                group_world,
            )
            return None
        return original(output, input_tensor, group=group, async_op=async_op)

    metadata_safe_all_gather._moe_metadata_gloo = True
    mappings.dist_all_gather_func = metadata_safe_all_gather
    logger.info("MOE_METADATA_GLOO_WRAP=ok")


def _install_a2a_split_trace():
    """Replace the wedged RCCL AllToAll kernel with an equivalent P2P ring."""
    import torch
    import torch.distributed as dist

    if getattr(dist.all_to_all_single, "_ep_split_trace", False):
        return

    original = dist.all_to_all_single
    call_count = 0

    def traced_all_to_all_single(*args, **kwargs):
        nonlocal call_count
        output = args[0] if len(args) > 0 else kwargs.get("output")
        input_tensor = args[1] if len(args) > 1 else kwargs.get("input")
        output_splits = (
            args[2] if len(args) > 2 else kwargs.get("output_split_sizes")
        )
        input_splits = (
            args[3] if len(args) > 3 else kwargs.get("input_split_sizes")
        )
        group = args[4] if len(args) > 4 else kwargs.get("group")
        async_op = args[5] if len(args) > 5 else kwargs.get("async_op", False)
        group_world = dist.get_world_size(group=group)
        should_route = (
            isinstance(output, torch.Tensor)
            and isinstance(input_tensor, torch.Tensor)
            and input_splits is not None
            and output_splits is not None
            and len(input_splits) == group_world
            and len(output_splits) == group_world
            and sum(input_splits) == input_tensor.shape[0]
            and sum(output_splits) == output.shape[0]
            and max([*input_splits, *output_splits]) <= 4096
            and group_world == dist.get_world_size()
        )
        if not should_route:
            return original(*args, **kwargs)

        if async_op:
            raise RuntimeError("ROCm EP P2P exchange requires async_op=False")

        transport_group = _safe_ep_group if _safe_ep_group is not None else group
        source_ranks = dist.get_process_group_ranks(group)
        transport_ranks = dist.get_process_group_ranks(transport_group)
        if source_ranks != transport_ranks:
            raise RuntimeError(
                f"EP transport ranks differ: source={source_ranks} "
                f"transport={transport_ranks}"
            )
        call_count += 1
        if call_count <= 4 or call_count % 100 == 0:
            logger.info(
                "EP_A2A_RING_P2P call=%d rank=%d source_group_rank=%d "
                "input_shape=%s output_shape=%s in=%s out=%s transport=safe_ep",
                call_count,
                dist.get_rank(),
                dist.get_rank(group=group),
                tuple(input_tensor.shape),
                tuple(output.shape),
                list(input_splits),
                list(output_splits),
            )

        group_rank = dist.get_rank(group=group)
        group_size = len(source_ranks)
        input_offsets = [0]
        output_offsets = [0]
        for count in input_splits:
            input_offsets.append(input_offsets[-1] + int(count))
        for count in output_splits:
            output_offsets.append(output_offsets[-1] + int(count))

        # One peer per batch is the only variant proven through a real optimizer step.
        # Wider windows pass synthetic traffic but deadlock on changing training splits.
        torch.cuda.synchronize()
        self_count = int(input_splits[group_rank])
        if self_count != int(output_splits[group_rank]):
            raise RuntimeError(
                f"self EP split differs: send={self_count} "
                f"recv={output_splits[group_rank]}"
            )
        output.narrow(0, output_offsets[group_rank], self_count).copy_(
            input_tensor.narrow(0, input_offsets[group_rank], self_count)
        )
        for first_step in range(1, group_size, 1):
            ops = []
            for step in range(first_step, min(first_step + 1, group_size)):
                send_peer = (group_rank + step) % group_size
                recv_peer = (group_rank - step) % group_size
                send_count = int(input_splits[send_peer])
                recv_count = int(output_splits[recv_peer])
                if send_count:
                    ops.append(
                        dist.P2POp(
                            dist.isend,
                            input_tensor.narrow(
                                0, input_offsets[send_peer], send_count
                            ),
                            peer=source_ranks[send_peer],
                            group=transport_group,
                            tag=step,
                        )
                    )
                if recv_count:
                    ops.append(
                        dist.P2POp(
                            dist.irecv,
                            output.narrow(
                                0, output_offsets[recv_peer], recv_count
                            ),
                            peer=source_ranks[recv_peer],
                            group=transport_group,
                            tag=step,
                        )
                    )
            if ops:
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
            torch.cuda.synchronize()
        return None

    traced_all_to_all_single._ep_split_trace = True
    dist.all_to_all_single = traced_all_to_all_single
    logger.info("EP_A2A_RING_P2P_WRAP=ok")


def _install_reload_warmup():
    """Warm each set of groups recreated after rollout."""
    import miles.backends.megatron_utils.actor as actor_module

    if getattr(actor_module.reload_process_groups, "_pg_reload_warmup", False):
        return

    original = actor_module.reload_process_groups

    def reload_process_groups_with_warmup(*args, **kwargs):
        result = original(*args, **kwargs)
        _warm_up_process_groups(None, install_hooks=False)
        # Build the transport communicator only after trainer memory and process groups
        # have been restored. Keeping an RCCL communicator across torch_memory_saver
        # pause/resume left 15 ranks stuck while rank 8 alone advanced in glm52x2ai.
        _prepare_safe_ep_group(force=True)
        logger.info("PG_RELOAD_WARMUP_DONE=ok safe_ep=fresh")
        return result

    reload_process_groups_with_warmup._pg_reload_warmup = True
    actor_module.reload_process_groups = reload_process_groups_with_warmup
    logger.info("PG_RELOAD_WARMUP_WRAP=ok target=actor.reload_process_groups")


def _run_forward_outside_memory_region():
    """Keep transient forward activations outside the pausable memory region."""
    try:
        from torch_memory_saver import torch_memory_saver

        from miles.backends.megatron_utils.actor import MegatronTrainRayActor

        original = MegatronTrainRayActor.compute_log_prob

        def compute_log_prob_outside_region(self, *args, **kwargs):
            with torch_memory_saver.disable():
                return original(self, *args, **kwargs)

        MegatronTrainRayActor.compute_log_prob = compute_log_prob_outside_region
        logger.info("PG_WARMUP_FWD_WRAP=ok target=MegatronTrainRayActor.compute_log_prob")
    except Exception as exc:
        logger.warning("PG_WARMUP_FWD_WRAP=failed %s", exc)
