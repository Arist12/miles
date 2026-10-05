"""Point-to-point fallback for the MoE dispatcher's all-to-all (``--moe-ep-p2p-alltoall``).

Across nodes, RCCL's AllToAll deadlocks on the dispatcher's uneven exchange even when every
rank's splits agree; the same split matrix moved as matched isend/irecv pairs, one peer at a
time, completes. Wider batches, and queuing every peer's batch before the first completes,
deadlocked as well. Use with --moe-token-counts-over-gloo: the ring trusts the splits, and a
rank whose splits disagree with its peers' waits forever.
"""

import logging
from collections.abc import Sequence
from itertools import accumulate

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def ring_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    group: dist.ProcessGroup | None = None,
) -> None:
    """``all_to_all_single`` along dim 0: step ``s`` sends to group rank ``r + s`` and receives
    from ``r - s``. The device is synchronized before the exchange and after every step, so a
    step never overlaps another step or another communicator's kernels."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    peers = dist.get_process_group_ranks(group) if group is not None else list(range(world))
    send = [int(n) for n in input_split_sizes]
    recv = [int(n) for n in output_split_sizes]
    send_at = [0, *accumulate(send)]
    recv_at = [0, *accumulate(recv)]
    sync = torch.cuda.synchronize if output.is_cuda else (lambda: None)

    sync()
    output.narrow(0, recv_at[rank], recv[rank]).copy_(input.narrow(0, send_at[rank], send[rank]))
    for step in range(1, world):
        dst, src = (rank + step) % world, (rank - step) % world
        ops = []
        if send[dst]:
            ops.append(dist.P2POp(dist.isend, input.narrow(0, send_at[dst], send[dst]), peers[dst], group))
        if recv[src]:
            ops.append(dist.P2POp(dist.irecv, output.narrow(0, recv_at[src], recv[src]), peers[src], group))
        if ops:
            for work in dist.batch_isend_irecv(ops):
                work.wait()
            sync()


class _RingAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.split_sizes = (output_split_sizes, input_split_sizes)
        input = input.contiguous()
        output = input.new_empty([sum(int(n) for n in output_split_sizes), *input.shape[1:]])
        ring_all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output_split_sizes, input_split_sizes = ctx.split_sizes
        grad_input = _RingAllToAll.apply(ctx.group, grad_output.contiguous(), input_split_sizes, output_split_sizes)
        return None, grad_input, None, None


def install() -> None:
    """Route the MoE token dispatcher's uneven all-to-alls through the ring."""
    from megatron.core.tensor_parallel import mappings
    from megatron.core.transformer.moe import token_dispatcher

    def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, **kwargs):
        if output_split_sizes_ is None or input_split_sizes is None or dist.get_world_size(group) == 1:
            return mappings.all_to_all(group, input_, output_split_sizes_, input_split_sizes, **kwargs)
        return _RingAllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)

    assert token_dispatcher.all_to_all is mappings.all_to_all, "the MoE dispatcher no longer uses mappings.all_to_all"
    token_dispatcher.all_to_all = all_to_all
    logger.info("MoE dispatcher all-to-alls go through the point-to-point ring")
