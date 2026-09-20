---
title: AMD ROCm
description: Run Miles on AMD MI350X / MI355X and MI300X / MI325X with the ROCm images. Docker is the recommended path.
---
Miles runs on AMD GPUs through ROCm. The ROCm images ship SGLang, Megatron-LM, and Miles
preinstalled, with `MILES_HARDWARE_PLATFORM=rocm` already set. The recipes and `train.py`
flags are the same as on NVIDIA; what changes is the image, the `docker run` flags, and the
launcher path.

## Images

The two `mi35x` images are built daily from `main` by the sgl-project/sglang nightly
workflows and published to Docker Hub under
[`rocm/sgl-dev`](https://hub.docker.com/r/rocm/sgl-dev/tags?name=miles):

| Image | ROCm | GPUs | Notes |
|---|---|---|---|
| `rocm/sgl-dev:miles-rocm10-mi35x` | 10 | MI350X / MI355X | Python 3.12 — the image the nightly tests run on |
| `rocm/sgl-dev:miles-rocm720-mi35x` | 7.2 | MI350X / MI355X | Python 3.10 |
| `rocm/sgl-dev:miles-rocm700-mi30x` | 7.0 | MI300X / MI325X | Not rebuilt daily — last built 2026-09-08 |

Each undated tag moves with every build; append `-YYYYMMDD` (e.g.
`miles-rocm10-mi35x-20260916`) to pin one.

To build an image yourself, `docker/Dockerfile.rocm` holds the recipe:

```bash
python docker/build.py --variant rocm10-mi35x --image-tag dev    # or rocm720-mi35x
```

## Start the container

On the **host**:

```bash
docker pull rocm/sgl-dev:miles-rocm10-mi35x

docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --shm-size 128G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it rocm/sgl-dev:miles-rocm10-mi35x /bin/bash
```

That drops you into a shell inside the container, with Miles at `/root/miles`, Megatron-LM
at `/root/Megatron-LM`, and SGLang at `/sgl-workspace/sglang`.

**Everything from here on runs inside the container.**

## Verify

Confirm Miles imports and the GPUs are visible:

```bash
python -c "import miles; print('Miles import OK')"
rocm-smi --showproductname
```

If either command fails, see [Debugging](/developer/debug).

## Launch training

The AMD launchers live under `scripts/amd/` and mirror the CUDA recipes. Download the model
and data and convert the checkpoint as in the [Quick Start](/getting-started/quick-start)
(Steps 2 and 3), then launch:

```bash
cd /root/miles
python scripts/amd/run_qwen3_4b.py --hardware MI355X    # or MI350X
```

The other recipes under `scripts/amd/` launch the same way.

## MI300X: FP32 → BF16 rounding in the AITER attention kernels

**If you are on MI300X / MI325X and see a much larger
`train/train_rollout_logprob_abs_diff` than the same run shows on MI350X / MI355X, this is
the first thing to check.**

AITER compiles its CK-tile attention kernels with a *truncating* FP32 → BF16 store.
`aiter/jit/optCompilerConfig.json` passes `-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` to every
`mha` module and defaults it to `2`, which is `ck_tile::bf16_rounding_mode::truncate`.
Truncation can only move a magnitude toward zero, so it biases every attention output
instead of adding symmetric noise, and the bias does not cancel across layers. Setting the
macro to `0` selects round-to-nearest-even.

The reason this shows up on MI300X and not MI350X is the dispatch order in
`aiter/ops/mha.py`. The forward tries, in order, a gfx950-only OPUS branch, then
`can_impl_fmha_v3_fwd() and seqlen_q > 128`, and only then falls through to the CK-tile
kernel. On gfx950 the OPUS branch absorbs the common dense BF16 shapes. On gfx942 those
shapes skip it, and the `seqlen_q > 128` guard sends every decode step and every short
extend chunk to the truncating kernel — while the trainer's full-sequence forward stays on
the ASM path. The two sides of the importance ratio then come from kernels that round
differently.

The Miles ROCm images set `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0` and drop the prebuilt
CK-tile `mha` modules so they rebuild under it. To go back to the upstream AITER behaviour,
rebuild with `--build-arg CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2`.

<Warning>

**Exporting the variable in a running container is usually not enough.** The macro is read
when a kernel is *compiled*, not when it is called. Any AITER module that is already
compiled — either shipped prebuilt in the image, or JIT-built earlier in the same container
— keeps the rounding it was built with, and setting the variable afterwards changes nothing
and reports nothing. `AITER_REBUILD` does not help either: the `mha` modules go through a
`gen_func`, which short-circuits that branch in `aiter/jit/core.py`.

To change it in a container you already have, delete the cached module and let it rebuild:

```bash
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
AITER_JIT=$(python3 -c 'import aiter, os; print(os.path.join(os.path.dirname(aiter.__file__), "jit"))')
rm -f  "$AITER_JIT"/mha_fwd_*.so "$AITER_JIT"/mha_varlen_fwd_*.so "$AITER_JIT"/mha_batch_prefill_*.so
rm -rf "$AITER_JIT"/build/mha_fwd_* "$AITER_JIT"/build/mha_varlen_fwd_* "$AITER_JIT"/build/mha_batch_prefill_*
```

The next attention call rebuilds them, which takes a minute or two per module.

</Warning>

Measured on one MI300X (gfx942), `rocm/sgl-dev:v0.5.18-rocm720-mi30x-20260826`, comparing
`aiter.flash_attn_func` on the CK-tile path against an FP64 reference over the same BF16
inputs:

| Build | Outputs biased toward zero | Mean abs relative error | µs / call |
|---|---|---|---|
| Default (truncate) | 89.5% | 0.0261 | 42.8 |
| `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0` | 50.2% | 0.0178 | 53.8 |

89.5% one-sided is the truncation signature; 50% is round-to-nearest. The ASM path the same
call takes at `seqlen_q > 128` already measures 50.1%, so the change makes the two paths
agree rather than making one of them unusual. Attention output error drops 32%, and the
kernel costs 26% more at this shape — a short-`seqlen_q` shape where the store is a large
share of the work, so treat that as closer to a worst case than to a step-time number.

End to end on Qwen3-8B, comparing rollout (decode) logprobs against the same tokens rescored
in one prefill pass, mean `|Δ logprob|` over 6144 tokens falls 9.1% on
`rlsys/miles:MI300-latest` and 4.9% on `rocm/sgl-dev:*-mi30x-*`. The older Miles image moves
more because its AITER applies the same truncating default to `module_rmsnorm`,
`module_rmsnorm_quant` and `module_fmha_v3_fwd` as well, which newer AITER no longer does.

## Next steps

- [Quick Start](/getting-started/quick-start) — the same Qwen3-4B run, step by step.
- [Hardware requirements](/getting-started/installation#hardware-requirements) — per-GPU status.
- [Low Precision RL](/advanced/low-precision) — FP8 block-wise on MI350X / MI355X.
- [Docker build](/developer/ci/02-docker-build) — the ROCm Dockerfile, variants, and tags.
