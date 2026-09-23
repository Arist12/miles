"""Ordered point-to-point replacement for the MoE expert-parallel all-to-all.

On ROCm 7.2 / RCCL 2.27.7, GLM-5.2 LoRA training under colocate deadlocks inside the
RCCL AllToAll kernel on the token-dispatch payload: every EP rank enqueues the same
collective on a fresh, warmed communicator and the kernel never completes. The same
split matrix moved as matched ``isend``/``irecv`` pairs, one peer at a time, completes.

``--moe-ep-p2p-alltoall`` turns this on. It then:

* serves every ``all_to_all_single`` on the expert-parallel ranks with
  :func:`ring_all_to_all_single` over a dedicated communicator (forward and
  ``_AllToAll.backward`` alike, since both reach ``torch.distributed.all_to_all_single``);
* moves the dispatcher's integer token-count all-gather on the world group to Gloo;
* owns that communicator's lifecycle across colocate sleep/wake: a pause makes it
  stale, and the next wake-up destroys it (trainer resident, nothing pausing
  underneath the teardown) and builds its replacement outside the memory-saver region,
  so one communicator exists at a time and none is used across a pause;
* connects the other Megatron process groups at the same point, also outside the
  memory-saver region, instead of lazily inside the first training forward.

The exchange must stay one peer per batch. Wider batches pass synthetic traffic with a
fixed split matrix but deadlocked on real training splits that change every layer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MAX_METADATA_NUMEL = 4096


def ring_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    *,
    peer_ranks: Sequence[int],
    my_index: int,
    transport_group: dist.ProcessGroup | None,
) -> None:
    """``all_to_all_single`` along dim 0 as a ring of matched isend/irecv pairs.

    ``peer_ranks`` are the global ranks of the all-to-all group in group order and
    ``my_index`` is this rank's position in it. Step ``s`` sends to ``my_index + s``
    and receives from ``my_index - s``; zero-sized chunks are skipped on both sides,
    which stays matched because the split matrix is consistent across ranks.
    """
    size = len(peer_ranks)
    assert len(input_split_sizes) == size and len(output_split_sizes) == size
    in_offsets = [0]
    for n in input_split_sizes:
        in_offsets.append(in_offsets[-1] + int(n))
    out_offsets = [0]
    for n in output_split_sizes:
        out_offsets.append(out_offsets[-1] + int(n))
    assert in_offsets[-1] == input.shape[0], (in_offsets[-1], input.shape)
    assert out_offsets[-1] == output.shape[0], (out_offsets[-1], output.shape)

    self_count = int(input_split_sizes[my_index])
    if self_count != int(output_split_sizes[my_index]):
        raise RuntimeError(
            f"EP self split differs: send={self_count} recv={int(output_split_sizes[my_index])}"
        )
    sync = torch.cuda.synchronize if output.is_cuda else (lambda: None)
    sync()
    output.narrow(0, out_offsets[my_index], self_count).copy_(input.narrow(0, in_offsets[my_index], self_count))

    for step in range(1, size):
        send_idx = (my_index + step) % size
        recv_idx = (my_index - step) % size
        send_count = int(input_split_sizes[send_idx])
        recv_count = int(output_split_sizes[recv_idx])
        ops = []
        if send_count:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    input.narrow(0, in_offsets[send_idx], send_count),
                    peer=peer_ranks[send_idx],
                    group=transport_group,
                )
            )
        if recv_count:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    output.narrow(0, out_offsets[recv_idx], recv_count),
                    peer=peer_ranks[recv_idx],
                    group=transport_group,
                )
            )
        if ops:
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        sync()


@dataclass
class _State:
    installed: bool = False
    ep_ranks: tuple[int, ...] = ()
    transport: dist.ProcessGroup | None = None
    stale: bool = False
    gloo_group: dist.ProcessGroup | None = None
    native_all_to_all_single: object = None
    calls: int = 0
    connected_groups: list[str] = field(default_factory=list)


_state = _State()


def _outside_memory_saver():
    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError:
        return nullcontext()
    return torch_memory_saver.disable()


def _raw_new_group(ranks: list[int]) -> dist.ProcessGroup:
    # A plain group, not a ReloadableProcessGroup: its lifetime is managed here, and
    # reload_process_groups() must not recreate it behind our back.
    new_group = getattr(dist, "old_new_group", dist.new_group)
    return new_group(ranks=ranks)


def _connect_megatron_groups() -> None:
    from megatron.core import parallel_state as mpu

    connected, seen = [], set()
    probe = torch.ones(8, dtype=torch.float32, device=torch.cuda.current_device())
    for name in sorted(n for n in dir(mpu) if n.startswith("get_") and n.endswith("_group")):
        try:
            group = getattr(mpu, name)()
        except Exception:
            # Getters that need arguments, or that assert when their feature is off.
            continue
        if not isinstance(group, dist.ProcessGroup) or id(group) in seen:
            continue
        if dist.get_backend(group) == "gloo":
            continue
        seen.add(id(group))
        try:
            dist.all_reduce(probe.clone(), group=group)
        except Exception as exc:
            logger.warning("could not connect %s ahead of use: %s", name, exc)
            continue
        connected.append(name)
    _state.connected_groups = connected

    ep_group = mpu.get_expert_model_parallel_group()
    ep_size = dist.get_world_size(group=ep_group)
    send = torch.zeros(ep_size * 8, dtype=torch.float32, device=torch.cuda.current_device())
    _state.native_all_to_all_single(torch.empty_like(send), send, group=ep_group)
    torch.cuda.synchronize()
    dist.barrier()


def _destroy_stale_transport() -> None:
    if _state.transport is None:
        return
    torch.cuda.synchronize()
    dist.barrier(group=_state.gloo_group)
    dist.destroy_process_group(_state.transport)
    _state.transport = None


def create_transport() -> None:
    """Build and connect the EP transport communicator (and the Megatron groups).

    Collective over the world: every rank calls it at the same point, with the trainer
    resident. A stale transport from before a pause is destroyed first; a live one is
    kept.
    """
    if not _state.installed or (_state.transport is not None and not _state.stale):
        return
    from megatron.core import parallel_state as mpu

    _destroy_stale_transport()
    ep_ranks = tuple(dist.get_process_group_ranks(mpu.get_expert_model_parallel_group()))
    memberships: list = [None] * dist.get_world_size()
    dist.all_gather_object(memberships, ep_ranks, group=_state.gloo_group)
    with _outside_memory_saver():
        _connect_megatron_groups()
        # Every rank creates every EP group, in the same order, as new_group requires.
        transport = None
        for ranks in sorted(set(memberships)):
            group = _raw_new_group(list(ranks))
            if ranks == ep_ranks:
                transport = group
        assert transport is not None
        # A tiny equal-split native all-to-all connects every peer in one launch; the
        # RCCL kernel only deadlocks on the real dispatch payload.
        send = torch.zeros(len(ep_ranks) * 8, dtype=torch.float32, device=torch.cuda.current_device())
        recv = torch.empty_like(send)
        _state.native_all_to_all_single(recv, send, group=transport)
        torch.cuda.synchronize()
    _state.ep_ranks = ep_ranks
    _state.transport = transport
    _state.stale = False
    logger.info(
        "EP P2P all-to-all transport ready: ep_ranks=%d connected_groups=%d",
        len(ep_ranks),
        len(_state.connected_groups),
    )


def mark_transport_stale() -> None:
    """The trainer is about to pause: stop using the transport, and let the next
    wake-up tear it down rather than destroying it underneath the pause."""
    _state.stale = True


def _is_ep_exchange(group) -> bool:
    if _state.transport is None or _state.stale:
        return False
    if dist.get_world_size(group=group) != len(_state.ep_ranks):
        return False
    if group is None:
        return _state.ep_ranks == tuple(range(dist.get_world_size()))
    return tuple(dist.get_process_group_ranks(group)) == _state.ep_ranks


def _wrap_all_to_all_single(original):
    def all_to_all_single(
        output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False
    ):
        # The dispatcher always passes explicit splits; anything else keeps the native path.
        if async_op or input_split_sizes is None or output_split_sizes is None or not _is_ep_exchange(group):
            return original(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=async_op,
            )
        size = len(_state.ep_ranks)
        _state.calls += 1
        if _state.calls == 1:
            logger.info("EP all-to-all served by the P2P ring (%d ranks)", size)
        ring_all_to_all_single(
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            peer_ranks=_state.ep_ranks,
            my_index=_state.ep_ranks.index(dist.get_rank()),
            transport_group=_state.transport,
        )
        return None

    all_to_all_single.__wrapped__ = original
    return all_to_all_single


def _wrap_metadata_all_gather(original):
    def all_gather_into_tensor(output, input, group=None, async_op=False):
        # Only the dispatcher's per-expert token counts: a small int64 vector.
        if (
            async_op
            or input.dtype != torch.int64
            or input.numel() > _MAX_METADATA_NUMEL
            or _state.gloo_group is None
            or dist.get_world_size(group=group) != dist.get_world_size()
        ):
            return original(output, input, group=group, async_op=async_op)
        world = dist.get_world_size()
        cpu_input = input.detach().cpu()
        cpu_outputs = [torch.empty_like(cpu_input) for _ in range(world)]
        dist.all_gather(cpu_outputs, cpu_input, group=_state.gloo_group)
        output.copy_(torch.cat(cpu_outputs, dim=0).to(output.device))
        return None

    all_gather_into_tensor.__wrapped__ = original
    return all_gather_into_tensor


def install(*, defer_transport: bool) -> None:
    """Route the EP all-to-all through the P2P ring. Call once, after model-parallel init.

    With ``defer_transport`` (colocate offload) the transport is first built at the
    first wake-up, so none exists before the first pause.
    """
    if _state.installed:
        return
    from megatron.core.tensor_parallel import mappings

    from miles.utils.distributed_utils import get_gloo_group

    _state.gloo_group = get_gloo_group()
    _state.native_all_to_all_single = dist.all_to_all_single
    dist.all_to_all_single = _wrap_all_to_all_single(dist.all_to_all_single)
    mappings.dist_all_gather_func = _wrap_metadata_all_gather(mappings.dist_all_gather_func)
    _state.installed = True
    if defer_transport:
        with _outside_memory_saver():
            _connect_megatron_groups()
    else:
        create_transport()


def is_installed() -> bool:
    return _state.installed
