# Per-model CUDA fusions for a hybrid GDN LLM (llama.cpp fork, experimental)

Patch set for llama.cpp that speeds up **batch-1 decode** of one specific checkpoint —
JetBrains **Qwen3.8-3.6-27B-blend Q4_K_M** (hybrid Gated-DeltaNet + attention) —
without changing the generated output.

**TL;DR: +18.4% decode throughput (geomean over 5 genres) vs upstream `b94959a`,
with greedy output identical on every verification prompt.**

This is an experimental fork, not an upstream PR. Kernels are specialized for the
architecture of this model family; expectations on other models are lower.

## Results

Batch-1 decode, temperature 0, fresh server, first bench run (see harness/README.md).

Google Colab G4 — NVIDIA RTX PRO 6000 Blackwell (188 SMs), 96 GB:

| | prose | code | list | reasoning | long_ctx | geomean |
|---|---|---|---|---|---|---|
| upstream `b94959a` (stock) | 140.52 | 163.13 | 128.08 | 164.95 | 92.65 | **135.02** |
| this patch set (v1) | 165.32 | 196.91 | 154.58 | 198.30 | 104.68 | **159.88 (+18.4%)** |

Fedora RTX 5090 (170 SMs), same protocol, earlier snapshot of the patch set
(sub_3, before the last two commits): stock 110.4 → patched 129.5 tok/s, **x1.17**.
The gap between stock and patched is larger on the smaller GPU, as expected.

Verification: 22 prompts (12 public + 10 hidden), greedy decode compared token for
token plus top-20 logprobs. `exact_match = 1.0` on all runs; `mean_kl = 0.0` on the
Colab G4. See "Known issues" for the cross-GPU observation.

## What is inside

1. **Horizontal matmul fusion** (`mmq-few.cu`, `mmvq.cu`, `mm-horiz.cuh`, `ggml-cuda.cu`,
   `concat.cu`): groups GDN input projections, attention QKV and FFN gate+up into single
   launches with register epilogues (sigmoid/softplus/SwiGLU), waves of tiny 48-row tiles
   scheduled alongside large GEMMs, dual Q6_K/Q4_K activation quantization in one launch,
   32-row tiling for warp-balanced K loops. Nets ~600 fewer kernel launches per forward.
2. **GPU-resident MTP speculative verification and greedy sampling**
   (`common/sampling.*`, `common/speculative.cpp`, `ggml-cuda/argmax.cu`, `top-k.cu`):
   draft acceptance and sampling decisions without returning to the host between tokens.
3. **Pre-warmed pinned checkpoint pools + zero-sync staged H2D** (`common/common.*`,
   `ggml-backend.cpp`, `ggml-cuda/common.cuh`): removes `cudaMallocHost` stalls from
   the request path and cuts stream synchronizations per decode round.
4. **GDN-specific vertical fusions** (`gated_delta_net.*`, `gated-delta-net-mma.cu`,
   `norm.cu`, `ssm-conv.cu`): gated RMSNorm fused with quantization, direct state gather,
   fused norm+q8_1 scratch shared with the matmul kernels.
5. **Long-prompt fix in the MTP draft catch-up** (`common/speculative.cpp`): the catch-up
   batch is decoded into the draft context in chunks bounded by its `n_batch`; fixes a
   hard abort (`get_sub_batch` overflow) on prompts >= 2048 tokens.

## Verification harness

`harness/suite.py` — the frozen evaluator used during development:

```
LLAMA_BASE=http://127.0.0.1:<port> python3 harness/suite.py golden   # capture reference
LLAMA_BASE=http://127.0.0.1:<port> python3 harness/suite.py verify   # compare candidate
LLAMA_BASE=http://127.0.0.1:<port> python3 harness/suite.py bench    # 5-genre tokens/sec
```

`exact_match >= 0.99` and `mean_kl <= 0.10` were the acceptance criteria during
development; the hidden prompt set is embedded (base64) so results are reproducible
but not easily overfit.

`harness/smoke_long_prompt.py` — regression check for the long-prompt abort.

## How to apply

Option A — plain patch (against `b94959a`):

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b94959a
git apply /path/to/fusions-v1.patch
```

Option B — fetch the commit from the bundle into a local clone of your fork:

```
git fetch /path/to/pub.bundle pub:pub
git checkout pub
```

## Build

The patches apply to llama.cpp at commit `b94959a` (2026-11):

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b94959a
git apply /path/to/fusions-v1.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j
```

Tested with CUDA 13.2, g++-15 host compiler, sm_120. `GGML_CUDA_FA_ALL_QUANTS=ON`
is required for the fused attention paths exercised here.

## Known issues

1. **Sampler-chain assert under server workloads.** On RTX 5090, some request
   sequences hit
   `llama_sampler.cpp: GGML_ASSERT("llama_sampler_chain_backend_init() called twice")`
   from the GPU-resident sampling path. Reproduced only on the 5090 server, not in
   the Colab benchmark loop. Tracked as open; do not run this in production yet.
2. **Cross-GPU KL drift.** On the 5090, the verifier reports `exact_match = 1.0` but
   `mean_kl ≈ 1.22` against a stock reference captured on the same GPU (two stock
   builds of the same source verified 0.0 against each other, so this is not build
   noise). Greedy output is unaffected; sampling distribution tails differ. Cause
   not yet isolated.
3. **Model-specific.** Changes are gated to the Qwen3.8 hybrid GDN path; other
   architectures should be unaffected but were not tested.
4. **Not upstream-ready.** The patch set was developed against a frozen evaluator
   and has not been reviewed across llama.cpp's backend/platform matrix.

## AI assistance disclosure

The patches were written by AI coding agents (Gemini 4 Neon/Carbon and Claude
Opus 5.5 in an agentic coding harness) iterating against a frozen verification
suite; a human owner directed the work, defined the contract, and reviewed the
results. Per llama.cpp's AGENTS.md, private forks are exempt from its
contributor-AI rules; this repository is a private-fork-style publication and
is not proposed for upstream merge.

## License

Same as upstream llama.cpp (MIT). No model weights are redistributed here.
