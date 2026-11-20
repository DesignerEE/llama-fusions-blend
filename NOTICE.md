# NOTICE

## Provenance

- Base: llama.cpp commit `b94959a` (2026-11), MIT.
- Generated on: Google Colab G4 (NVIDIA RTX PRO 6000 Blackwell, 188 SMs, 96 GB,
  CUDA 13.2, g++-15, sm_120) and validated on Fedora (RTX 5090, 170 SMs, sm_120).
- Model used for all numbers: `Qwen3.8-3.6-27B-blend-Q4_K_M.gguf`
  (JetBrains). The weights are NOT included in this repository.

## How the code was produced

The patch set was produced by AI coding agents (Gemini 4 Neon, Gemini 4 Carbon,
Claude Opus 5.5) inside an agentic coding harness, iterating against a frozen
verification suite that the agents could not modify. The human owner defined the
task, the acceptance contract and the evaluation protocol, and reviewed every
submission's hypothesis and result. This matches the emerging "autoresearch"
pattern (agents search kernel space; a frozen evaluator guards correctness).

llama.cpp's own AGENTS.md prohibits autonomous PR submissions and AI-written
PR text; this repository is a private-fork-style publication of patches and
harnesses, not a pull request, and is not proposed for upstream merge.

## Contents map

- `patches/v6.patch` — full diff against `b94959a`: fusions (34 files, +3639/−242)
  plus the v6 sampler lifecycle fix (+4 lines in `tools/server/server-context.cpp`).
- `patches/fusions-v1.patch` — the fusions-only snapshot, kept for reference.
- `bundle/llama-fusions-blend.bundle` — git bundle (`b94959a..v6`) with the
  `v6` tag, for fetching into a local clone of llama.cpp.
- `bundle/v1.bundle` — the earlier fusions-only bundle.
- `docs/RESULTS.md` — the development journal: six submissions, per-genre
  numbers, nsys profiles, ablation results (including negative ones), and the
  root-cause write-up of the MTP draft-context overflow.
- `harness/` — the frozen evaluator and the long-prompt regression smoke.

## Known issues (see README)

1. Sampler-chain assert (`llama_sampler_chain_backend_init() called twice`) —
   FIXED in v6: the detached greedy-persist chain is dropped so backend init
   runs once per chain. Verified by a request-cycle smoke (greedy -> logit_bias
   -> greedy) on RTX 5090, no abort.
2. Cross-GPU logprob drift (exact-match identical; KL differs) — open.
3. Model-family specific (Qwen3.8 hybrid GDN path).
