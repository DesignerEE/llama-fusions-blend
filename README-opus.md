```
  ####  #  #   ###  ###   ##   #  #   ###
  #     #  #  #      #   #  #  ## #  #
  ###   #  #   ##    #   #  #  # ##   ##
  #     #  #     #   #   #  #  #  #     #
  #      ##   ###   ###   ##   #  #  ###
  ========================================================================
  llama-fusions-blend :: per-model CUDA fusions for llama.cpp, batch-1 decode
```

A per-model patchset for **llama.cpp `b94959a`** that speeds up single-stream greedy decoding of
**JetBrains Qwen3.8-3.6-27B-blend (Q4_K_M)**. The code is CUDA and was tuned on `sm_120`.

```
+- TL;DR -----------------------------------------------------------------
| +18.4% batch-1 decode geomean   135.02 -> 159.88 tok/s   (RTX PRO 6000, 188 SMs)
| x1.17 on an RTX 5090 snapshot   reported with the same protocol, not re-measured here
| On the G4 box, the full output (reasoning tokens included) matched stock token-for-token
| Our old verifier's exact_match=1.0 was partly vacuous; see "Verification" below
| Experimental private-fork patchset. Not proposed upstream.
+-------------------------------------------------------------------------
```

## Results

Protocol: start a fresh server, run `harness/suite.py verify`, then `harness/suite.py bench`, and
keep the **first** bench run only. Both servers use the same fixed CLI flags (below).
`bench` sends 5 prompts with `temperature=0` and `max_tokens=192`. It reports
`tok/s = completion_tokens / wall time`, so prompt processing is included.

```
 genre       stock   v6      x             0         50        100       150       200 tok/s
 ---------   ------  ------  -----         |---------|---------|---------|---------|
 prose       140.52  165.32  1.176   stock ############################
                                     v6    #################################
 code        163.13  196.91  1.207   stock #################################
                                     v6    #######################################
 list        128.08  154.58  1.207   stock ##########################
                                     v6    ###############################
 reasoning   164.95  198.30  1.202   stock #################################
                                     v6    ########################################
 long_ctx     92.65  104.68  1.130   stock ###################
                                     v6    #####################
 ---------   ------  ------  -----
 geomean     135.02  159.88  1.184   (1 '#' = 5 tok/s)
```

| milestone | what it adds | geomean tok/s | vs stock |
|---|---|---|---|
| stock `b94959a` | baseline | 135.02 | 1.000x |
| sub_2 (`455dd40`) | horizontal matmul + norm/launch-boundary fusion | 144.66 | 1.071x |
| sub_3 (`3cd00e9`) | GPU-resident MTP/sampling, GDN state gather, pinned pool | 151.98 | 1.126x |
| sub_4 (`c507be2`) | zero-sync H2D ring, pre-warmed checkpoints, norm+q8_1 fusion | 160.08 | 1.186x |
| **v6** (sub_6) | MTP `n_batch` overflow fix (and the sampler fix, see Known issues) | **159.88** | **1.184x** |

These are `nsys` measurements of one speculative round, where a round processes 1 sampled token
plus 3 draft tokens:

| | stock | sub_4 |
|---|---|---|
| time per round | 20.317 ms | 17.287 ms |
| kernel launches per round | 2,576 | 1,169 |
| SM busy (188 SMs) | 0.837 | 0.913 |

`long_ctx` gains the least because its ~3.6k-token prompt prefill (~0.85 s) counts toward wall time.

## What's inside

CUDA files are under `ggml/src/ggml-cuda/`; other paths are relative to the llama.cpp root.

| theme | what changes | main files |
|---|---|---|
| Matmul fusion | gate+up+SwiGLU in one kernel (`mul_mat_q_few_glu`); one launch for small matrices sharing an input (`mul_mat_q_few_horiz_kernel`) | `mmq-few.cu`, `mm-horiz.cuh`, matchers in `ggml-cuda.cu` |
| Norm / elementwise | residual add + RMSNorm + q8_1 quantisation (`add_rms_norm_mul_quant`); fused GDN output gate (`rms_norm_gated_silu_128_quant`) | `norm.cu`, `norm.cuh`, `rope.cu`, `concat.cu` |
| Gated DeltaNet | recurrent state read in place (`state_src_rows`, no gather copy); PDL launches | `gated_delta_net.cu`, `gated-delta-net-mma.cu`, `src/models/qwen35.cpp` |
| MTP + sampling on GPU | draft hidden-state rows stay on the device; deferred catch-up decodes; GPU top-k/argmax; v6 decodes drafts in `n_batch` chunks | `common/speculative.cpp`, `top-k.cu`, `argmax.cu`, `src/llama-context.cpp` |
| Host / launch overhead | 2x2 MiB pinned H2D staging ring, no stream sync; CUDA-graph-safe q8_1 scratch; pinned checkpoint buffers allocated at startup | `ggml-backend.cpp`, `common.cuh`, `common/common.cpp`, `tools/server/server-context.cpp` |

```
 one decode step (1 sampled + 3 MTP drafts, CUDA graph, PDL between kernels)

 tokens --> [embed] --> +---------------- 64 trunk layers (48 GDN, 16 FA) -----------------+
                        | [add+RMSNorm+q8_1] -> [horizontal QKV/gate mm] -> [GDN | FlashAttn] |
                        | [gated RMSNorm+silu+q8_1] -> [out proj] -> [add+RMSNorm+q8_1]      |
                        | [gate+up+SwiGLU fused mm] -> [down proj]                           |
                        +--------------------------------------+-----------------------------+
                                                               |
             [LM head] --> [argmax on GPU] --> accept/verify <-+
                 |
                 +--> [MTP head x3] --> [top_k_small on GPU] --> drafts --> next step
```

## How to apply

Start from a clean checkout of `b94959a`, then use either the patch or the bundle (one commit per milestone):

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b94959a
git apply /path/to/llama-fusions-blend/patches/v6.patch                                   # option 1
git fetch /path/to/llama-fusions-blend/bundle/llama-fusions-blend.bundle v6:fusions-v6   # option 2
```

## Build

We built with CUDA 13.0.88 and GCC 13.3.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_QUANTS=all
cmake --build build --target llama-server -j
```

All runs used the same flags (from `scripts/start_server.sh`), and they were never tuned:

```
llama-server -m Qwen3.8-3.6-27B-blend-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 999 -fa on -c 204800 -np 1 -ctk q4_0 -ctv q4_0 -b 2048 -ub 1024 \
  --kv-unified --cache-ram 90112 --cache-reuse 256 -t 24 --threads-batch 24 --jinja
```

## Verification, and what we got wrong

The evaluator `harness/suite.py` was frozen and checked against `MANIFEST.sha256` before every run.
It compares each candidate against a "golden" output captured from a stock build on the same machine.

1. **The old `exact_match` was partly vacuous.** `suite.py` compares only `message.content`. This is a
   reasoning model, and in 11 of the 12 public verify prompts all 64 generated tokens go to
   `reasoning_content`. So for those prompts `content` is `""` on both sides, and an empty string
   always equals an empty string.
2. **The corrected contract compares `reasoning_content + content` token by token.** We re-ran this on
   the G4 box with a separate dev probe: sub_2, sub_3, sub_4 and v6 all produce **exactly the same
   tokens as stock** (764 positions across the 12 public prompts, top-20 logprobs identical). Stock
   is also deterministic, run to run and against its own golden output.
3. **KL measures noise as much as error.** The suite computes KL position by position. Once two
   runs pick a different token, every later position compares different contexts, so the KL grows
   even when nothing is wrong. Stock against stock shows this:

   | change vs stock golden | mean_kl | KL before the first different token | where the first difference starts |
   |---|---|---|---|
   | `--spec-draft-n-max 2` instead of 3 | 0.473 | 3.1e-4 | top-2 gaps of 0.0006 to 0.058 nats |
   | speculation off | 0.450 | 3.1e-4 | top-2 gaps of 0.0006 to 0.14 nats |

   A change that only reorders fp32 additions (an experimental split-K matmul, with matmul outputs
   within 1e-6 of stock) gives a mean_kl of 0.51 to 0.67 in the same way.
4. **The RTX 5090 observation.** On the 5090, sub_3 scored `exact_match=1.0` with `mean_kl=1.22`
   against a stock golden output. Two stock builds there scored KL 0.0 against each other. This fits
   the pattern above: a tiny numeric difference flips one near-tie, then the outputs diverge. It is
   **not a proven numeric error**, but we have not explained it either. The corrected token-level
   contract has not been run on the 5090 yet.

**Recommended gate:** exact token match over `reasoning_content + content`. Use KL only as a
diagnostic, and only on the positions before the first token where the two outputs differ.

## Known issues

- **Sampler lifecycle crash: fixed in v6.** sub_3 keeps one persistent greedy sampler chain per
  slot. On a mix of requests (greedy, then a request with logprobs, then greedy again) the server
  re-attached that same chain, and `llama_sampler_chain_backend_init` asserted "called twice". v6
  frees the detached chain so each chain is initialised only once (`tools/server/server-context.cpp`).
  This fix landed after the benchmark run above.
- **MTP batch overflow on prompts of 2048+ tokens: fixed in sub_6/v6.** `get_sub_batch` aborted with
  `n = 2051 > n_batch` when queued catch-up draft tokens were added to a full prompt chunk. The draft
  decode is now split into `n_batch` chunks. The regression test is `harness/smoke_long_prompt.py`.
- **KL metric caveat**: see above. Do not read a KL number on its own as a correctness verdict.
- **Specific to one model family.** The matchers target the `qwen35` hybrid graph (48 Gated DeltaNet
  + 16 full-attention layers, 1 MTP layer) with Q4_K/Q6_K weights, a q4_0 KV cache, batch 1 and
  `sm_120`. No other model or GPU has been tested.
- **Not ready for upstream.** It has env-var knobs and per-model heuristics, and one fused path
  (`ssm_conv_state_fused_f32`) is disabled because it caused KL drift. No maintainer has reviewed it.

## AI-assistance disclosure

Most of the code, profiling and measurement was done by AI coding agents (Gemini 4 Neon/Carbon and
Claude Opus 5.5) working in a sandbox under human direction. The evaluator was frozen and
hash-checked, the stock server was never touched, and every number here comes from a `verify.json`,
`bench_run1.json` or `nsys` artifact. This is a private-fork-style experiment published for
reference. It is not proposed upstream.

## Repo layout

```
llama-fusions-blend/
|-- patches/      v6.patch (b94959a -> v6), per-milestone diffs sub_2 .. sub_6
|-- bundle/       llama-fusions-blend.bundle (git history, one commit per milestone)
|-- harness/      suite.py + MANIFEST.sha256 (frozen evaluator), smoke_long_prompt.py
|-- results/      verify.json, bench_run1.json and hypothesis.md for each milestone
|-- scripts/      start_server.sh (fixed flags)
`-- docs/         REPORT.md (full engineering log, profiles, ablations)
```

## License

MIT, the same as llama.cpp. The model weights are not included and have their own license.

## References and inspiration

- Megakernel-per-model write-up: https://dangxingyu.github.io/megakernel
- lucebox (lucemegakernel): per-model persistent-kernel inference
- AutoMegaKernel: automated per-model kernel fusion

These projects aim at one persistent kernel per model. This repo stops earlier: it uses graph-level
fusion inside stock llama.cpp, and does not have a persistent megakernel yet.

```
Status: experimental
```
