#!/usr/bin/env python3
"""Long-prompt regression smoke for the MTP draft catch-up fix.

Sends prompts of increasing size through an OpenAI-compatible llama-server
and reports success/failure per size. Intended to catch the historical
`get_sub_batch: failed to add token ... (n = 2049..2051)` abort with prompt
chunks >= the model's -b value.

usage: python3 smoke_long_prompt.py [base_url] [token_sizes_csv]
       python3 smoke_long_prompt.py http://127.0.0.1:1239 3000,10000,30000,60000
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1239"
SIZES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "3000,10000,30000,60000").split(",")]


def ask(n_tokens: int) -> bool:
    filler = ("The quick brown fox jumps over the lazy dog. " * (n_tokens // 9 + 2))[: n_tokens * 5]
    body = json.dumps({
        "model": "blend-q4",
        "messages": [{"role": "user", "content": filler + "\nReply with one word: ok?"}],
        "temperature": 0,
        "max_tokens": 16,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=300)
        print("prompt~%dtok: OK %.1fs" % (n_tokens, time.time() - t0))
        return True
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        print("prompt~%dtok: FAIL %s" % (n_tokens, str(e)[:120]))
        return False


if __name__ == "__main__":
    ok = all(ask(n) for n in SIZES)
    sys.exit(0 if ok else 1)
