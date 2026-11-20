# REPORT — Qwen3.8-3.6-27B-blend Q4_K_M on RTX PRO 6000 Blackwell (sm_120)

## Run 1 (recovered from `/content/mk/refs/journal_run1.md`, pre-VM-crash)

### sub_1 — no-op pipeline: PASS
- Сборка стока в свой build-каталог, кандидат-сервер `:1240`.
- `verify`: PASS, `exact_match 1.0`, `mean_kl 0.0` (614 позиций) — движок детерминирован.
- `bench` честного прогона (свежий сервер): `geomean 134.89 = 1.021×` dispatch-бейзлайна = пол шума (±1%).
- Прозой `+9.6%` в первом прогоне — warmup-эффект базы (первый запрос после старта).

### Ловушка измерения (поймана, протокол принят)
- `bench` run2 на том же сервере: `geomean 153.60 = 1.163×` — НЕЗАЧЁТ: `--cache-ram` кэширует промпт, `long_ctx` пересчитывал 4 токена вместо 3626.
- Протокол: свежий кандидат → `verify` → `bench` run1 = официальное число; парное сравнение с no-op-сборкой на тех же условиях; `:1239` не трогать.

### Профиль sub_1 (`nsys`, `--cuda-graph-trace=node`, prose)
- Forward = 2405 кернелей; PDL даёт ~0.7 мкс start-to-start на мелких.
- `mul_mat_q_few` (few-row int8-MMA MMQ, `ncols=4`): 12.3 мс = 78% прохода, 15 GB весов за проход ⇒ 1.22 TB/s производственного стриминга.
- `quantize_mmq_few` 0.41 мс, GDN 0.30, concat 0.34, `rms_norm` 0.26, FA 0.20.
- 496 матмулов/forward × ~5.5 мкс фиксированных ≈ 2.7 мс non-streaming.
- Мелочь на нищих гридах: `ssm_alpha`/`beta` 48 строк = 3-блочный грид.

### Руфлайн (собственные инструменты, `dev/qbench` + `bw.cu`)
- `test-backend-ops perf` ВРЁТ: веса L2-resident ⇒ фиктивные 2.75 TB/s.
- `qbench` (40 различных копий весов, форс DRAM): `17408×5120` Q4_K `n=4`: 38.0 мкс — совпадает с производством (39.4).
- `bw.cu`: асимптота 1.535 TB/s; ~2-3 мкс на границу запуска.
- ВЫВОД: `mmq_few` уже в 3-10% от per-launch руфлайна на больших матрицах. Оставшийся рычаг = границы запусков/хвосты/мелкие матмулы ⇒ horizontal fusion.

### Потолок (арифметика владельца)
- `1.22 → 1.535 TB/s` (`×1.26`) + 2.7 мс фикс на ~15.6 мс прохода (~17%) ⇒ потолок ≈ `×1.5`.
- GOLD `1.35×` достижим только ступенью 4 (persistent kernel, CPU-free steady state).

### Что НЕ нужно делать заново
- `sub_1` конвейер доказан; профиль и руфлайн сняты (числа выше).
- Quantize dedup ≈ 0.2 мс/forward (~1%) — не стоит отдельного сабмита.
- В апстриме на `b12557` уже есть: few-row MMQ (Q4_K dequant в регистрах + int8 MMA), gate/up/GLU MMVQ-фьюжн (`ncols=1`), `rms_norm+mul`, `rope+set_rows`, PDL.

---

## Run 2 (rebuilt VM, 2026-11-20)

### Baseline capture on rebuilt VM (`:1239`, `golden` → `bench` run1)
- `golden_public` (12) + `golden_hidden` (10) captured against stock `:1239`; `MANIFEST.sha256` verified OK.
- Baseline `bench` run1 (after `golden` warmup, matching `verify → bench run1` protocol):
  - `prose`: **140.52** tok/s | `code`: **163.13** tok/s | `list`: **128.08** tok/s | `reasoning`: **164.95** tok/s | `long_ctx`: **92.65** tok/s
  - **geomean: 135.02 tok/s** (matches Run 1 `sub_1` no-op `134.89` within `0.1%`).
  - Pass bar (`1.05×` all genres & geomean): geomean ≥ **141.77**, `prose` ≥ 147.55, `code` ≥ 171.29, `list` ≥ 134.48, `reasoning` ≥ 173.20, `long_ctx` ≥ 97.28.
  - Tiers (`TARGETS.md`): BRONZE (`1.15×`) = **155.27** | SILVER (`1.25×`) = **168.78** | GOLD (`1.35×`) = **182.28**.

### sub_2 — horizontal matmul + residual/gated-RMSNorm + launch-boundary fusion: PASS (`1.071×` geomean, `mean_kl = 0.0`)
- Branch/dir: `/content/submissions/sub_2/` (`patch.diff`, own build `/content/submissions/sub_2/build`).
- What changed:
  1. **4-way GDN horizontal matmul + epilogue fusion** (`wqkv + z + ssm_beta + ssm_alpha` + `sigmoid(beta)` + `softplus(alpha + dt) * a` across `mmq_few` `ncols=4..8` and `mmvq` `ncols=1..3`, with single-launch dual `DS4`/`D4` quantization on mixed `Q6_K+Q4_K` layers; packs 3-CTA `beta`/`alpha` grids into Wave 0; `11 → 2` launches/layer × 48).
  2. **3-way Full-Attention QKV horizontal matmul fusion** (`wq + wk + wv`, `6 → 2` launches/layer × 17 incl. MTP).
  3. **FFN `gate + up + SwiGLU` fusion for `mmq_few` (`ncols=4..8`) & multi-col `mmvq` (`ncols=2..5`)** (`5 → 2` launches/layer × 64, preserving stock IEEE-754 warp K-reduction order).
  4. **Fused `ADD(residual) + RMS_NORM + MUL`** (`130×`/round), **fused GDN `RMS_NORM + MUL(ssm_norm) + SiLU(z)*MUL`** (`48×`/forward), **joint Q+K `L2_NORM`** (`96 → 48` launches), **2D tiled + PDL `concat_non_cont`** (`10,240 → 320` CTAs), **PDL on `ssm_conv_long_token` / `gdn`**, and **per-decode-batch CUDA Graph bucketing (`ncols=1..8`)**.
- Numerical lesson caught & solved: reassociating `Q4_K` FP32 K-sums on short-prompt prefill (`ncols=9..24`) can flip a near-tied `<think>` token and spike position-zipped KL; keeping stock IEEE-754 K-order and restricting `mmq_few` fusions to `ncg=1` (`ncols=4..8`, 100% of decode) yields **`exact_match = 1.0, mean_kl = 0.0`** across both all 12 public prompts (764 pos) and all 10 hidden prompts (614 pos). Also caught `CUDA_SET_SHARED_MEMORY_LIMIT` caching a smaller dynamic smem cap when a `Q4_K`-only layer launched before a `Q6_K+Q4_K` layer.
- Official fresh-server evaluation (`:1240` → `verify` → `bench` run1):
  - `verify`: **PASS** (`exact_match: 1.0`, `mean_kl: 0.0`, `positions: 614`).
  - `bench` run1 vs baseline (`135.02`):
    - `prose`: **151.10** tok/s (`1.075×`) | `code`: **175.71** tok/s (`1.077×`) | `list`: **140.20** tok/s (`1.095×`) | `reasoning`: **175.77** tok/s (`1.066×`) | `long_ctx`: **96.83** tok/s (`1.045×`, pure decode = **182.24** tok/s, `916 ms` / `1980 ms` = 3,626-tok prefill)
    - **geomean: 144.66 tok/s (`1.071×` baseline — clears `1.05×` TARGETS.md pass bar)**.
- Server slot profile insight (`server.log`): pure decode is `5.09–6.64 ms/tok` (`~18–20.5 ms`/speculative round), while ~`2.8–3.2 ms`/round (~15% wall time) is lost to CPU `top_k(10)`/`argmax` over `248,320` floats (`7 MB`/round D2H logits) + synchronous `h_nextn` D2H/H2D round-trips in `draft-mtp`. Next (`sub_3` / `sub_4`): baseline `nsys` SM busy metric, eliminate all CPU/PCIe sampling & MTP state round-trips on GPU, and push persistent/fused execution to BRONZE (`1.15×`) → SILVER (`1.25×`) → GOLD (`1.35×+`).

### Stock `nsys` SM Busy Baseline (`TARGETS.md` Req 3, `:1240` stock build, `prose` steady-state 40 rounds)
- Trace: `/content/mk/nsys/stock_prose.nsys-rep` (`--trace=cuda --cuda-graph-trace=node`).
- Steady-state speculative round (`40` rounds = `812.68 ms` → **`20.317 ms/round`**, `2,576` kernels/round):
  - **Steady-state GPU kernel-active fraction**: **`0.9206`** (`1.61 ms/round` GPU completely idle across 5 host/D2H gaps: `292.9 µs` post-trunk, `586.8 µs` CPU 4-token verify sampling, `148.5 µs` + `142.2 µs` + `309.5 µs` CPU MTP `top_k(10)` & `h_nextn` D2H/H2D; `4.07 MB/round` D2H memcpys).
  - **Steady-state effective SM busy fraction (188 SMs, wave-occupancy weighted)**: **`0.8365`** (`< 0.85` target).
  - **Single trunk forward (`2,404` kernels, `15.647 ms`)**: GPU kernel-active **`0.9967`**, effective SM busy **`0.9017`**.
  - Top non-trunk kernel: MTP draft LM head `mul_mat_vec_q<Q6_K, ncols=1>` (`248,320` blocks) = `682.9 µs` × 3 = `2.05 ms/round` (`8.5%` of total GPU time).

### sub_3 — GPU-resident speculative MTP/sampling + GDN direct-state gather + pinned checkpoint pool: PASS (`1.126×` geomean, `mean_kl = 0.0`)
- Branch/dir: `/content/submissions/sub_3/` (`patch.diff`, own build `/content/submissions/sub_3/build`).
- What changed (cumulative on top of `sub_2`):
  1. **GPU-resident greedy target sampling & O(1) acceptance**: auto-enabled backend `argmax` (`float4` + PDL, deterministic smallest-ID tie-break) with persistent `smpl_greedy_persist` when `temp <= 0 && n_probs == 0`, eliminating `4.0 MB/round` D2H raw logits and 4× CPU vocab scans (`16 B` D2H instead of `4.0 MB`).
  2. **GPU-resident MTP draft sampling (`top_k_small_f32`) + device-only `h_nextn` + merged `ctx_dft` catch-up**: 2-stage deterministic zero-atomic CUDA `top_k(10)` (`80 B` D2H instead of `1.0 MB/step`), 8-slot `h_nextn_dev` D2D stash with device event waits, preserved `pos0` KV in `ctx_dft`, and merged `0..3` accepted catch-up rows into Step 0 of `draft()`.
  3. **GDN direct `ssm_states_all` gather (`gdn_get_rows_fusion`) & shared `s_slot_rev`**: expanded `q_conv`/`k_conv`/`v_conv` before `build_rs(ssm_states_all)` and replaced `ggml_can_fuse_subgraph` (which falsely rejected non-constant KV views) with `ggml_cuda_can_elide_to_single_output`, activating `GET_ROWS + GATED_DELTA_NET + CPY` fusion on all 48 GDN layers (`302 MB/forward` VRAM traffic eliminated, `-12..18 ms/request` pure decode); `ensure_rs_slot_rev()` builds `s_slot_rev` once per graph (`141` fewer launches).
  4. **FA strided gate view (`19` `cpy_scalar`/round eliminated) + `mmvq` `rows_per_block = 2` for `Q4_K`/`Q6_K` `ncols=1`**.
  5. **Pinned `common_state_data` pool (`batch-copy.cu` `cudaPointerGetAttributes` skip of `cudaHostRegister`/`cudaHostUnregister`) + zero graph reallocations (`0 / 0`)**: moved `dummy_reserve_smpl` release before the final `pp` reserve in `sched_reserve()` and re-reserved `galloc` once in `set_sampler()`.
- **Ablation summary (closed — `PUB` 8 prompts / 512 positions & `HID` 10 prompts / 614 positions)**:
  - `MROPE`/`IMROPE` fused `rms_norm_mul_rope`: disabled after ablation showed 1-ULP FMA reassociation across 16 attn layers × ~45 `<think>` tokens flipped 1 `<think>` token (`mean_kl = 0.4325` → `0.000000` when disabled).
  - `ssm_conv_state_fused_f32` (`ablate.log`, `graph reallocs = 0 / 0` across all 4 cases):
    - `ALL_ON`: `exact = 8/8`, `mean_kl = 0.412308` (`HID` `mean_kl = 0.164`, `exact_match = 1.0`, draft acceptance dropped `86.5% → 64.1%`).
    - `NO_GDN_FUSE`: identical `mean_kl = 0.412308` (exonerates `gdn_get_rows_fusion`).
    - `NO_PIN_POOL`: identical `mean_kl = 0.412308` (exonerates pinned state pool & `batch-copy.cu`).
    - `NO_CONV_FUSE` (`gdn_get_rows_fusion` ON + `common_pinned_state_pool` ON): **`exact = 8/8`, `mean_kl = 0.000000`, `max_kl = 0.000000`** (`pos0` exact to 6 decimals on all 8 prompts).
  - PDL launch-completion lesson: calling `ggml_cuda_pdl_lc()` at kernel entry in `top_k`/`argmax`/`ssm_conv` released non-PDL downstream `k_get_rows` under CUDA Graph replay on `sm_120`, causing illegal memory access; removed `ggml_cuda_pdl_lc()` from those kernels.
- Official fresh-server evaluation (`:1240` → `verify` → `bench` run1):
  - `verify`: **PASS** (`exact_match: 1.0`, `mean_kl: 0.0`, `positions: 614`).
  - `bench` run1 vs baseline (`135.02`) and `sub_2` (`144.66`):
    - `prose`: **161.36** tok/s (`1.148×`, pure decode `176.85`) | `code`: **187.47** tok/s (`1.149×`, pure decode `204.90`) | `list`: **145.75** tok/s (`1.138×`, pure decode `158.64`) | `reasoning`: **184.83** tok/s (`1.121×`, pure decode `208.32`) | `long_ctx`: **99.50** tok/s (`1.074×`, pure decode `193.47`)
    - **geomean: 151.98 tok/s (`1.1256×` baseline, `+5.06%` over `sub_2`; ALL 5 genres ≥ `1.074×`, `graph reallocs = 0 / 0`)**.
  - Gap to BRONZE (`155.27 tok/s = 1.15×`): only **`+3.29 tok/s` (`+2.16%`)** remaining. Next: `nsys` profile of `sub_3` → `sub_4` targeting BRONZE (`1.15×`) and SILVER (`1.25×`).

- **`sub_3` `nsys` Steady-State Profile (`/content/submissions/sub_3/nsys_sub3.sqlite`, `prose` 40 rounds)**:
  - Steady-state round: **`17.678 ms/round`** (down from stock `20.317 ms/round`), **`1,392` kernels/round** (down from `2,576`), D2H traffic **`0.25 KB/round`** (down from `4,070 KB/round`), GPU kernel-active **`0.9423`**, effective SM busy (188 SMs) **`0.7659`**, trunk segment `13.935 ms` with `0` inter-kernel gaps `> 3 µs` and `2.83 ms` PDL overlap.
  - **Top Remaining Bottlenecks Identified for `sub_4`**:
    1. **64+ Synchronous `cudaStreamSynchronize` Calls in `ggml_backend_sched_copy_input` (`4` gaps = `861 µs/round`, `~350–450 µs` sync overhead)**: Every `GGML_TENSOR_FLAG_INPUT` tensor (`8/step × 4 steps = 32/round`) calls `ggml_backend_synchronize(split_backend)` + synchronous `cudaMemcpyAsync(..., cudaStreamPerThread)` + `cudaStreamSynchronize(cudaStreamPerThread)`.
    2. **Split-K Wave Tail on `128×` `nrows=5120` `mul_mat_q_few` (`blk = 320` on `188` SMs = `1.70` waves, `56` SMs idle for 50% of kernel; `4,220 µs/round` = `24.0%` of round)**: `ffn_down` Q6_K (`60.1 µs` vs `47.6 µs` roofline), `ffn_down` Q4_K (`~43 µs` vs `32.7 µs`), `ssm_out` Q4_K (`15.2 µs` vs `7.7 µs`), `wo` Q4_K (`~16.5 µs` vs `11.5 µs`) — `~0.95–1.1 ms/round` headroom + directly drags down SM busy (`0.7659`).
    3. **Lazy `cudaMallocHost` Stalls During Bench Requests (`5× cudaMallocHost = 122.1 ms` in first request + additional checkpoint misses in `list`/`reasoning`)**: `common_pinned_state_pool` allocated `151.6 MB` pinned buffers lazily on demand.
    4. **Trunk CUDA Graph Chunking (`8× cudaGraphLaunch = 680 µs` CPU launch overhead, `91 µs` on critical path)**: default `chunk_size = 512` splits the trunk into 8 separate graphs.

## sub_4 — BRONZE Cleared (`160.08 tok/s = 1.186×`, `mean_kl = 0.0`)
- **Commit / Branch**: `sub_4` (worktree `/content/submissions/sub_4/src`, patch `/content/submissions/sub_4/patch.diff`).
- **What Changed**:
  1. **Zero-Sync Staged Async H2D Input Copy** (`ggml-backend.cpp`, `common.cuh`, `ggml-cuda.cu`): added double-buffered `2 × 2 MiB` pinned host staging ring in `ggml_backend_cuda_context` and routed `GGML_TENSOR_FLAG_INPUT` host tensors (`<= 1 MiB`) through `ggml_backend_tensor_set_async` on `split_backend`'s compute stream. Cut `cudaStreamSynchronize/round` from `107 → 40` and total host gaps from `861 µs → 553 µs/round` (`-308 µs/round`).
  2. **Pre-Warmed Pinned Checkpoint Pool** (`common.h`, `common.cpp`, `server-context.cpp`): pre-allocated `64 × 192 MiB`, `16 × 64 MiB`, and `64 × 2 MiB` pinned slabs once at server startup (before `/health` OK) with `32 MiB` alignment and `free_list` cap `128`. Eliminated `2–3 × 21 ms` `cudaMallocHost` stalls per request across `verify` and all 5 `bench` prompts (`code`/`list`/`reasoning` prompt eval dropped `89–100 ms → 47–57 ms`; `long_ctx` dropped `922 ms → 853 ms`).
  3. **Fused Norm + Pre-Quantized `q8_1` Scratch & Active GDN Gated RMSNorm** (`common.cuh`, `norm.cuh`, `norm.cu`, `mmq-few.cu`, `ggml-cuda.cu`): static CUDA-Graph-safe `prequant_ds4`/`prequant_d4` scratch on `cuda_ctx`; consumer-aware `add_rms_norm_mul_quant_f32_kernel` (`127×/trunk`, single-layout when consumer only needs `ds4` or `d4`); fixed `ggml_cuda_match_rms_norm_gated_silu` (`ggml_cuda_can_elide_to_single_output` instead of `ggml_can_fuse_subgraph` which rejected `GGML_OP_VIEW(z)`) activating `rms_norm_gated_silu_128_quant_f32_kernel` on all 48 GDN layers (`-181 µs/round` wall).
  4. **Negative Ablations Reverted**: `GGML_CUDA_GRAPH_CHUNK_SIZE=0` (`-0.3 to -0.9 tok/s` pure decode vs `512` due to critical-path launch latency) and fused Split-K `nwarps=2, nunits_stage=1` (`-2.5 tok/s` cold + `27` reallocs + KL `0.34`) were reverted cleanly.
- **Official Fresh-Server Eval (`verify.json`, `bench_run1.json`)**:
  - `verify`: **`{"exact_match": 1.0, "mean_kl": 0.0, "positions": 614, "verdict": "PASS"}`**
  - `prose`: **165.64 tok/s** (`1.179×`, pure decode `178.64`, `0/0` reallocs)
  - `code`: **197.25 tok/s** (`1.209×`, pure decode `207.32`, `0/0` reallocs)
  - `list`: **154.78 tok/s** (`1.208×`, pure decode `160.60`, `0/0` reallocs)
  - `reasoning`: **198.39 tok/s** (`1.203×`, pure decode `210.63`, `0/0` reallocs)
  - `long_ctx`: **104.79 tok/s** (`1.131×`, pure decode `196.19`, `0/0` reallocs)
  - **Geomean: `160.08 tok/s` (`1.1856×` baseline `135.02`, `+5.33%` over `sub_3` — BRONZE `155.27` cleared by `+4.81 tok/s`)**
- **`nsys` Steady-State Profile (`/content/submissions/sub_4/nsys_sub4.sqlite`)**:
  - `17.287 ms/round` (30 steady-state rounds; 1 full round `17.166 ms`), `1,168.8` kernels/round (`-223` vs `sub_3`), GPU kernel-active **`0.9545`**, effective SM busy (188 SMs) **`0.9125`**, `4` host gaps = `553.2 µs/round`, `40× cudaStreamSynchronize/round`.
  - Remaining levers for `sub_5` (targeting SILVER `168.78 tok/s` = `1.25×`, `+5.4%` from `160.08`):
    1. Remaining `40× cudaStreamSynchronize` + `553 µs/round` host gaps (`81920 B` / `20480 B` `strm=15` sync H2D + `4` host round-trips → GPU-resident MTP loop / async `set_inputs`).
    2. `mul_mat_q_few [blk=320]` (`3,923.5 µs` wall = `28.5%` of trunk) — standalone Split-K `K_split=2, nwarps=4` (keeping `nunits_stage=2` on `Q4_K` and `nstages=3` on `Q6_K`, zero graph reallocs) specifically on `ssm_out` (`48×`), `wo` (`16×`), and `ffn_down` (`64×`).
    3. `add_rms_norm_mul_quant [blk=8]` (`522.1 µs` wall) & `quantize_mmq_few_q8_1 [blk=272]` (`183.7 µs` wall) & GDN conv `k_get_rows + concat + set_rows + ssm_conv` (`280 µs` wall).
    4. `long_ctx` prompt eval (`852.9 ms` = `46.6%` of `long_ctx` wall time) and short prompt eval (`47–64 ms`).


## sub_6 — Production Hotfix: MTP Draft `n_batch` Overflow (`159.88 tok/s = 1.184×` BRONZE, `mean_kl = 0.0`, Long-Prompt Smoke PASS)
- **Commit / Branch**: `304187d` (`sub_6` branched from `sub_4` `c507be2`; full patch `/content/submissions/sub_6/patch.diff`, surgical 1-file fix `/content/submissions/sub_6/fix_only.diff`; folded into in-flight `sub_5` as well).
- **Production Bug & Root Cause**:
  - Fedora RTX 5090 prod (`sub_3` / `094f893`) aborted on long prompts (`>= 2048` tokens) with `common/common.cpp:2304: get_sub_batch: failed to add token <id> at index 2048 (error -1, n = 2051)`.
  - Reproduced identically on unfixed `sub_4` (`common_speculative_impl_draft_mtp::process` -> `common_batch::get_sub_batch`, `n = 2050`/`2051`).
  - Cause: `sub_3`'s GPU-resident MTP catch-up deferral stores up to `n_draft_max` (`3`) accepted draft tokens in `catchup_q[seq_id]`. When a subsequent prompt reuses the cached prefix ending at those tokens and sends a full `n_batch = 2048` chunk (`batch_in.size() == 2048`), `common_speculative_impl_draft_mtp::process()` prepended `1..3` `catchup_q` tokens onto `batch` before appending all `2048` `batch_in` tokens (`batch.size() = 2049..2051 > llama_n_batch(ctx_dft) = 2048`), then called `batch.get()` (`get_sub_batch(0, batch.size())`), overflowing `ctx_dft`'s `llama_batch_ext` capacity at index `2048`.
- **Fix (`common/speculative.cpp`)**:
  - In `common_speculative_impl_draft_mtp::process()`, decode `batch` into `ctx_dft` in chunks bounded by `n_batch_dft = (int32_t) llama_n_batch(ctx_dft)` via `batch.get_sub_batch(off, n_cur)` (`n_cur = std::min(n_batch_dft, n_dft_total - off)`). Preserves causal order and absolute `embd_state_row` device indices into read-only `ctx_tgt` `h_nextn_dev`; zero overhead when `batch.size() <= n_batch_dft`.
- **Regression Smoke (`/content/mk/smoke_long_prompt.py`, outside frozen `/content/mk/verify/`)**:
  - Added to eval loop (`sub_6/run_eval.sh` & `sub_5/run_eval.sh`): cold ~3900-token prompt, 4× exact `catchup_q` + `2048`-token chunk continuations (`max_tokens = 40, 41, 42, 43`, ~4370 tokens), multi-turn chat (~4191 tokens), and fresh ~4135-token prompt after speculative generation. **All 7 cases PASS (`9.9s`)**.
- **Official Fresh-Server Eval (`/content/submissions/sub_6/verify.json`, `/content/submissions/sub_6/bench_run1.json`)**:
  - `verify`: **`{"exact_match": 1.0, "mean_kl": 0.0, "positions": 614, "verdict": "PASS"}`**
  - `prose`: **165.32 tok/s** | `code`: **196.91 tok/s** | `list`: **154.58 tok/s** | `reasoning`: **198.30 tok/s** | `long_ctx`: **104.68 tok/s**
  - **Geomean: `159.88 tok/s` (`1.1841×` baseline `135.02`, BRONZE, `0/0` graph reallocs across all prompts)**
  - Baseline `:1239` health OK; `/content/mk/verify/MANIFEST.sha256` OK.
