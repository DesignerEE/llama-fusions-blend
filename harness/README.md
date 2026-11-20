# Harness

Three files, all runnable against any OpenAI-compatible llama-server:

- `suite.py` — the frozen evaluator: `golden` captures reference outputs
  (greedy text + top-20 logprobs) from the stock build; `verify <base_url>`
  compares a candidate (`exact_match >= 0.99`, `mean_kl <= 0.10` were the
  development gates); `bench` measures tokens/sec over five genres.
- `smoke_long_prompt.py` — regression check for the long-prompt abort
  (prompt chunks >= `-b` plus MTP draft catch-up).
- `RESULTS.md` (in `docs/`) — development journal with per-submission numbers,
  nsys profiles and the root-cause write-up of the draft-context overflow.

## Protocol

All numbers in the README were measured with this protocol:

1. Start a FRESH server (cold prompt cache) with the same flags for stock and
   candidate. Flags used: `-ngl 999 -fa on -c 32768 -np 1 -ctk q4_0 -ctv q4_0
   -b 2048 -ub 1024 --jinja --spec-type draft-mtp --spec-draft-n-max 3`.
2. `suite.py golden` on the stock build once (or `verify` against an existing
   golden set).
3. `suite.py verify` on the candidate — must pass before benchmarking.
4. `suite.py bench` — the FIRST bench run on a fresh server is the official
   number. Re-runs on a warm `--cache-ram` server overstate results (prompt
   cache hits make `long_ctx` re-evaluate almost nothing).
