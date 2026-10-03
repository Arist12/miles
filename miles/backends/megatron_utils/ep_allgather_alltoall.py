"""All-gather replacement for the MoE dispatcher's uneven all-to-all (``--moe-alltoall-via-allgather``).

Across nodes, RCCL's point-to-point path intermittently loses a receive completion: its
AllToAll deadlocks on the dispatcher's uneven exchange, and so does the same split matrix
moved as matched isend/irecv pairs (about once per 20 rollouts). Collectives over the same
group are unaffected, so the exchange is two all-gathers: the split matrix, then every rank's
send buffer padded to the longest one. Each rank keeps its own slices.
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist


def allgather_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    group: dist.ProcessGroup | None = None,
) -> None:
    """``all_to_all_single`` along dim 0, moving world x the data of the busiest rank."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    splits = torch.tensor([int(n) for n in input_split_sizes], dtype=torch.int64, device=input.device)
    gathered_splits = splits.new_empty(world * world)
    dist.all_gather(list(gathered_splits.chunk(world)), splits, group=group)
    matrix = gathered_splits.view(world, world).tolist()  # matrix[src][dst] in group order
    assert [row[rank] for row in matrix] == [int(n) for n in output_split_sizes], "inconsistent split sizes"

    rows = max(sum(row) for row in matrix)
    send = input.new_zeros((rows, *input.shape[1:]))
    send[: input.shape[0]] = input
    gathered = input.new_empty((world * rows, *input.shape[1:]))
    dist.all_gather(list(gathered.chunk(world)), send, group=group)

    at = 0
    for src, row in enumerate(matrix):
        n = row[rank]
        output.narrow(0, at, n).copy_(gathered.narrow(0, src * rows + sum(row[:rank]), n))
        at += n


class _AllGatherAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.split_sizes = (output_split_sizes, input_split_sizes)
        input = input.contiguous()
        output = input.new_empty([sum(int(n) for n in output_split_sizes), *input.shape[1:]])
        allgather_all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output_split_sizes, input_split_sizes = ctx.split_sizes
        grad_input = _AllGatherAllToAll.apply(
            ctx.group, grad_output.contiguous(), input_split_sizes, output_split_sizes
        )
        return None, grad_input, None, None


def install() -> None:
    """Route the MoE token dispatcher's uneven all-to-alls through all-gathers."""
    from megatron.core.tensor_parallel import mappings
    from megatron.core.transformer.moe import token_dispatcher

    def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, use_nccl_stream=False):
        if output_split_sizes_ is None or input_split_sizes is None or dist.get_world_size(group) == 1:
            return mappings.all_to_all(group, input_, output_split_sizes_, input_split_sizes, use_nccl_stream)
        return _AllGatherAllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)

    token_dispatcher.all_to_all = all_to_all
