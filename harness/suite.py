
import json, time, urllib.request, base64, os, sys

BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:1239")
V = os.path.dirname(os.path.abspath(__file__))

def chat(messages, max_tokens=192, temp=0, logprobs=False, top=20, stream=False):
    body = {"model": "blend-q4", "messages": messages, "temperature": temp,
            "max_tokens": max_tokens, "stream": stream}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

BENCH = {
 "prose": [{"role":"user","content":"Write one tight paragraph (about 120 words) explaining why the sky is blue."}],
 "code": [{"role":"user","content":"Write a Python function quicksort(lst) with docstring, no external libs."}],
 "list": [{"role":"user","content":"List 10 practical tips for writing maintainable Python code, numbered."}],
 "reasoning": [{"role":"user","content":"A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? Reason step by step, then answer."}],
 "long_ctx": [{"role":"user","content": ("Here is a log excerpt:\n" + "\n".join(f"2026-11-{d:02d} 12:0{n}:0{m} service=api latency={120+n*7+d}ms status={200 if n%3 else 503}" for d in range(1,26) for n in range(6) for m in range(3))[:6000]) + "\n\nHow many 503 statuses are in the log, and on which dates do they cluster? Answer briefly."}],
}

# public equivalence prompts (visible in this file)
PUB = [
 "Continue exactly: The capital of France is",
 "Translate to Russian: The early bird catches the worm.",
 "def fib(n):\n    # complete this function\n",
 "2+2*2 = ? Answer with the number only.",
 "Name the four largest planets in order.",
 "Write a haiku about compilers.",
 "What is 15% of 240? Answer with the number only.",
 "List three sorting algorithms with O(n log n) average complexity.",
 "Continue: In the beginning of the 20th century, physics was transformed by",
 "Сколько дней в феврале в невисокосный год? Ответ числом.",
 "Reverse the string 'llama.cpp' without explanation.",
 "What HTTP status code means 'Not Found'? Number only.",
]
# hidden prompts: base64, still in this frozen file; agent rule: no reading beyond bench calls
_H = "WyJDb250aW51ZTogVGhlIHNwZWVkIG9mIGxpZ2h0IGluIHZhY3V1bSBpcyAiLCAiU3RhdGUgdGhlIGVxdWF0aW9uIGZvciB0aGUgYXJlYSBvZiBhIGNpcmNsZS4iLCAiV2hhdCBpcyB0aGUgY2hlbWljYWwgc3ltYm9sIGZvciBnb2xkPyIsICJQcm9jZWVkIHRvIEVuZ2xpc2g6INCa0YPQtNCwINC40LTRkdC8INC80YsuINCf0LvRj9GI0LXQvC4iLCAiV3JpdGUgYSB3b3JkIHRoYXQgcmh5bWVzIHdpdGggJ2NvcmUnLiIsICJIb3cgbWFueSBiaXRzIGluIGEgYnl0ZT8gTnVtYmVyIG9ubHkuIiwgIkNvbnRpbnVlOiBUaGUgbW9zdCBpbXBvcnRhbnQgc3RlcCB5b3UgdGFrZSB3aGVuIGRlYnVnZ2luZyBhIHB1YmxpYyBsaWJyYXJ5IGlzIiwgIldoYXQgZG9lcyBzcyBzdGFuZCBmb3IgaW4gVU5JWD8gT25lIHdvcmQuIiwgIk5hbWUgdGhlIHByaW1lIG51bWJlciBhZnRlciA3NTYzLiIsICLQn9C10YDQtdCy0LXQtNC4INC90LAg0LvQsNGC0LjQvdC40YbRgzog0KHRh9Cw0YHRgtC70LjQstC+0LPQviDQutC+0LTQuNC90LPQsC4g0JjQu9C4INC90LXRgi4iXQ=="
HID = json.loads(base64.b64decode(_H))

def bench():
    res = {}
    for genre, msgs in BENCH.items():
        t0 = time.time()
        r = chat(msgs, max_tokens=192, temp=0, stream=False)
        wall = time.time() - t0
        ch = r["choices"][0]
        usage = r.get("usage", {})
        nct = usage.get("completion_tokens", len(ch.get("message",{}).get("content","").split()))
        ttft_hdr = r.get("_ttft", None)
        res[genre] = {"tok_s": round(nct / wall, 2) if wall > 0 else 0,
                      "completion_tokens": nct, "wall_s": round(wall, 2)}
        print(f"[bench] {genre}: {nct} tok in {wall:.2f}s = {res[genre]['tok_s']} tok/s")
    geo = 1.0
    for g in res.values(): geo *= max(g["tok_s"], 0.01)
    res["geomean"] = round(geo ** (1/len(res)), 2)
    json.dump(res, open(os.path.join(V, "last_bench.json"), "w"), indent=1)
    return res

def capture(tag, prompts):
    out = []
    for i, p in enumerate(prompts):
        r = chat([{"role":"user","content":p}], max_tokens=64, temp=0, logprobs=True)
        ch = r["choices"][0]
        entry = {"i": i, "text": ch.get("message",{}).get("content","")}
        lp = (ch.get("logprobs") or {}).get("content")
        if lp:
            entry["toks"] = [{"t": e["token"], "p": e["top_logprobs"][0]["logprob"] if e.get("top_logprobs") else None,
                              "top": [(t["token"], t["logprob"]) for t in (e.get("top_logprobs") or [])[:20]]} for e in lp]
        out.append(entry)
    json.dump(out, open(os.path.join(V, tag + ".json"), "w"), ensure_ascii=False)
    print(f"[capture] {tag}: {len(out)} prompts, logprobs={'yes' if out and 'toks' in out[0] else 'NO'}")
    return out

def verify(cand_base):
    globals()["BASE"] = cand_base
    import math
    hid = json.load(open(os.path.join(V, "golden_hidden.json")))
    agree, kls, n = 0, [], 0
    for e in hid:
        r = chat([{"role":"user","content":HID[e["i"]]}], max_tokens=64, temp=0, logprobs=True)
        ch = r["choices"][0]
        txt = ch.get("message",{}).get("content","")
        lp = (ch.get("logprobs") or {}).get("content") or []
        gold_txt = e["text"]
        if txt.strip() == gold_txt.strip(): agree += 1
        for pe, ge in zip(lp, e.get("toks", [])):
            if ge.get("top") and pe.get("top_logprobs"):
                gt = {t: p for t, p in ge["top"]}
                ct = {x["token"]: x["logprob"] for x in pe["top_logprobs"]}
                if ct:
                    kl = sum(math.exp(p) * (p - ct.get(t, -20.0)) for t, p in gt.items() if t in ct)
                    kls.append(max(kl, 0.0)); n += 1
    token_agree = agree / len(hid)
    mean_kl = (sum(kls) / len(kls)) if kls else None
    verdict = "PASS" if token_agree >= 0.99 and (mean_kl is None or mean_kl <= 0.10) else "FAIL"
    print(json.dumps({"exact_match": round(token_agree,4), "mean_kl": round(mean_kl,4) if mean_kl is not None else None, "positions": n, "verdict": verdict}))
    return verdict

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode == "bench": bench()
    elif mode == "golden":
        capture("golden_public", PUB)
        capture("golden_hidden", HID)
    elif mode == "verify":
        verify(sys.argv[2] if len(sys.argv) > 2 else BASE)
