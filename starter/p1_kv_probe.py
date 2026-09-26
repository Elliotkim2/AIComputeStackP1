#!/usr/bin/env python3
"""
p1_kv_probe.py -- Project 1 starter, "Transformer Dissection".

CSE 40702, The AI Compute Stack.

RUNS ON A LAPTOP.  CPU, Apple Silicon (MPS), or CUDA, whatever you have.
No cluster account required.  On a 2020-or-later laptop the default run takes
two to four minutes.

WHAT THIS DOES
--------------
  A. Exact byte accounting for one forward pass: predicted KV cache from the
     formula, measured KV cache, measured logits, everything else.  The four
     numbers add up.                                   -> handout Task 1
  B. Attention concentration vs. sequence length, per layer.  -> Task 2
  C. Next-token distribution vs. sampling parameters.          -> Task 3
  Task 4 (extrapolation) has no script part; it is done by hand.

It does NOT write your report and it does NOT explain what it finds.  That is
the assignment.

USAGE
-----
    pip install "transformers>=5.0" torch
    python p1_kv_probe.py                      # both assigned models, defaults
    python p1_kv_probe.py --models openai-community/gpt2 --seqlens 128 512
    python p1_kv_probe.py --csv p1_group07     # write the raw artifacts

The two default models are chosen so that one uses multi-head attention and
the other uses grouped-query attention.  The same formula should work for both,
and it only does if you use the right head count.

NOTATION (the same symbols as L06 and the handout)
--------------------------------------------------
    L = layers   H = query heads   H_kv = key/value heads   d_k = head dim
    n = tokens   u = concurrent sequences (batch)   b = bytes per number
    KV bytes = 2 * L * H_kv * d_k * n * u * b
    The JSON keys below are spelled out (layers, heads_q, heads_kv, head_dim)
    because they mirror the config file; the line under them prints the same
    numbers in the handout's symbols.

THREE THINGS THAT COULD COST YOU POINTS
---------------------------------------
 1. The naive KV formula uses QUERY heads (H).  A grouped-query model needs
    KEY/VALUE heads (H_kv).  If your measured/predicted ratio comes out at
    exactly 2, 4, or 8, that is why.
 2. FlashAttention and the fused attention path never build the n x n score
    matrix, so asking for attention weights returns nothing usable.  This
    script loads with attn_implementation="eager" for part B.  Say in your
    report that you did, and what it costs.
 3. The KV cache is usually NOT the biggest tensor in a forward pass.  Part A
    shows you what is.  Do not report the wrong number as "the cache."
"""

import argparse
import csv
import json
import math
import platform
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# One MHA model and one GQA model.  Both are small, ungated, and download in
# well under a minute on a normal connection.
DEFAULT_MODELS = ["openai-community/gpt2", "Qwen/Qwen3-0.6B"]

REAL_TEXT = (
    "Mike ran a cross country race at Northridge and it was hot. The course "
    "climbed for the first mile, flattened along the ridge, and then dropped "
    "back through the trees to the finish. Data centers have a similar "
    "problem: the peak load is not the average load, and the cooling system "
    "has to be sized for the peak. "
) * 40


# --------------------------------------------------------------------------
# device and dtype
# --------------------------------------------------------------------------
def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device):
    # fp32 on CPU: half precision on CPU is slow and sometimes unsupported.
    return torch.float32 if device.type == "cpu" else torch.float16


def device_label(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


# --------------------------------------------------------------------------
# the formula under test
# --------------------------------------------------------------------------
def model_facts(cfg):
    L = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    hq = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    hkv = getattr(cfg, "num_key_value_heads", None) or hq
    dh = getattr(cfg, "head_dim", None) or (hidden // hq)
    vocab = getattr(cfg, "vocab_size", None)
    return dict(layers=L, heads_q=hq, heads_kv=hkv, head_dim=dh,
                hidden=hidden, vocab=vocab, gqa_ratio=hq / hkv,
                attention="MHA" if hq == hkv else f"GQA {hq}:{hkv}")


def predicted_kv_bytes(f, seq_len, batch, elem_bytes, use_query_heads=False):
    """Equation 1.  Set use_query_heads=True to reproduce the WRONG version."""
    heads = f["heads_q"] if use_query_heads else f["heads_kv"]
    return 2 * f["layers"] * heads * f["head_dim"] * seq_len * batch * elem_bytes


# --------------------------------------------------------------------------
# A. exact byte accounting for one forward pass
# --------------------------------------------------------------------------
def tensor_bytes(x):
    return x.numel() * x.element_size() if torch.is_tensor(x) else 0


def cache_bytes(past):
    """Sum every tensor inside a transformers Cache object or legacy tuple."""
    if past is None:
        return 0
    found = []
    if hasattr(past, "layers"):                     # transformers v5
        for layer in past.layers:
            for name in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, name, None)
                if torch.is_tensor(t):
                    found.append(t)
    if not found:                                   # older Cache API
        for name in ("key_cache", "value_cache"):
            seq = getattr(past, name, None)
            if seq is not None:
                found += [t for t in seq if torch.is_tensor(t)]
    if not found:                                   # legacy tuple of tuples
        try:
            for entry in past:
                for t in entry:
                    if torch.is_tensor(t):
                        found.append(t)
        except TypeError:
            pass
    return sum(tensor_bytes(t) for t in found)


def measure_forward(model, facts, seq_lens, device, dtype, batch, vocab):
    elem = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for L in seq_lens:
        ids = torch.randint(0, vocab, (batch, L), device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        kv = cache_bytes(out.past_key_values)
        logits = tensor_bytes(out.logits)
        other = 0
        for k, v in out.items() if hasattr(out, "items") else []:
            if k in ("past_key_values", "logits"):
                continue
            if torch.is_tensor(v):
                other += tensor_bytes(v)
            elif isinstance(v, (list, tuple)):
                other += sum(tensor_bytes(t) for t in v)

        pred_ok = predicted_kv_bytes(facts, L, batch, elem, False)
        pred_bad = predicted_kv_bytes(facts, L, batch, elem, True)

        rows.append(dict(
            seq_len=L, batch=batch,
            predicted_kv_bytes=pred_ok,
            predicted_kv_bytes_using_query_heads=pred_bad,
            measured_kv_bytes=kv,
            logits_bytes=logits,
            other_output_bytes=other,
            total_returned_bytes=kv + logits + other,
            ratio_measured_over_predicted=(kv / pred_ok) if pred_ok else float("nan"),
            logits_over_kv=(logits / kv) if kv else float("nan"),
            wall_seconds=round(wall, 3),
        ))
        del out
        free_cache(device)
    return rows


def free_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------
# B. attention concentration
# --------------------------------------------------------------------------
def measure_attention(model, tok, seq_lens, device, text, topq=0.05,
                      second_half_only=True):
    rows = []
    enc = tok(text, return_tensors="pt")
    for L in seq_lens:
        if enc["input_ids"].shape[1] < L:
            print(f"  (skipping n={L} for part B: the sample text tokenizes to "
                  f"{enc['input_ids'].shape[1]} tokens, which is shorter)")
            continue
        ids = enc["input_ids"][:, :L].to(device)
        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)
        if not out.attentions:
            raise RuntimeError("no attention weights returned -- reload with "
                               "attn_implementation='eager'")
        for li, A in enumerate(out.attentions):        # (1, heads, L, L)
            A = A[0].float()
            lo = L // 2 if second_half_only else 0     # skip trivially-short rows
            A = A[:, lo:, :]
            k = max(1, int(topq * L))
            top = A.topk(k, dim=-1).values.sum(-1)
            p = A.clamp_min(1e-12)
            ent = -(p * p.log()).sum(-1)
            rows.append(dict(
                seq_len=L, layer=li, top_fraction=topq,
                rows_scored=("second half" if second_half_only else "all"),
                mean_top_mass=float(top.mean()),
                mean_entropy_nats=float(ent.mean()),
                max_entropy_nats=math.log(L),
                normalized_entropy=float(ent.mean()) / math.log(L),
                effective_keys=float(torch.exp(ent.mean())),
                first_position_mass=float(A[:, :, 0].mean()),
            ))
        del out
        free_cache(device)
    return rows


# --------------------------------------------------------------------------
# C. sampling
# --------------------------------------------------------------------------
def measure_sampling(model, tok, prompts, device,
                     temperatures=(0.1, 0.5, 0.7, 1.0, 1.5, 2.0)):
    rows = []
    for pi, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        for T in temperatures:
            p = torch.softmax(logits / T, dim=-1)
            srt, idx = p.sort(descending=True)
            cum = srt.cumsum(0)
            H = float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())
            rows.append(dict(
                prompt_index=pi, prompt=prompt[:60], temperature=T,
                top1_prob=float(srt[0]), top1_token=tok.decode(idx[0]),
                entropy_nats=H, effective_vocab=math.exp(H),
                n_tokens_for_90pct=int((cum < 0.90).sum()) + 1,
                n_tokens_for_99pct=int((cum < 0.99).sum()) + 1,
            ))
    return rows


# --------------------------------------------------------------------------
def run_one(model_id, args, device, dtype):
    print("\n" + "=" * 78)
    print(f"  {model_id}")
    print("=" * 78)

    cfg = AutoConfig.from_pretrained(model_id)
    facts = model_facts(cfg)
    print("ARCHITECTURE " + json.dumps(facts))
    print(f"  in the handout's notation: L = {facts['layers']}, "
          f"H = {facts['heads_q']}, H_kv = {facts['heads_kv']}, "
          f"d_k = {facts['head_dim']}, d_model = {facts['hidden']}, "
          f"|V| = {facts['vocab']}")
    if facts["gqa_ratio"] > 1:
        print(f"  This model uses {facts['attention']}. Equation 1 needs "
              f"H_kv = {facts['heads_kv']}, not H = {facts['heads_q']}. "
              f"Using the wrong one over-predicts by "
              f"{facts['gqa_ratio']:.0f}x.")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager").to(device).eval()

    elem = torch.tensor([], dtype=dtype).element_size()
    print(f"\n--- A. byte accounting for one forward pass (dtype={args.dtype}, "
          f"{elem} B/elem) ---")
    kv = measure_forward(model, facts, args.seqlens, device, dtype,
                         args.batch, facts["vocab"])
    hdr = (f"{'n':>6} {'pred KV':>10} {'meas KV':>10} {'ratio':>7} "
           f"{'logits':>10} {'other':>9} {'total':>10} {'sec':>7}")
    print(hdr)
    for r in kv:
        print(f"{r['seq_len']:>6} {r['predicted_kv_bytes']/2**20:>10.2f} "
              f"{r['measured_kv_bytes']/2**20:>10.2f} "
              f"{r['ratio_measured_over_predicted']:>7.3f} "
              f"{r['logits_bytes']/2**20:>10.2f} "
              f"{r['other_output_bytes']/2**20:>9.2f} "
              f"{r['total_returned_bytes']/2**20:>10.2f} "
              f"{r['wall_seconds']:>7.2f}")
    print("        (all byte columns in MiB)")

    att = []
    if not args.skip_attention:
        alens = [n for n in args.seqlens if n <= args.attn_max_len]
        print(f"\n--- B. attention concentration (n <= {args.attn_max_len}, "
              f"scoring the second half of each sequence) ---")
        att = measure_attention(model, tok, alens, device, REAL_TEXT)
        print(f"{'n':>6} {'layer':>6} {'top5% mass':>11} {'H/Hmax':>8} "
              f"{'eff keys':>9} {'pos-0 mass':>11}")
        for r in att:
            print(f"{r['seq_len']:>6} {r['layer']:>6} {r['mean_top_mass']:>11.3f} "
                  f"{r['normalized_entropy']:>8.3f} {r['effective_keys']:>9.1f} "
                  f"{r['first_position_mass']:>11.3f}")

    print("\n--- C. sampling ---")
    samp = measure_sampling(model, tok, args.prompts, device)
    print(f"{'#':>3} {'T':>5} {'p(top1)':>9} {'H':>7} {'eff vocab':>10} "
          f"{'n@90%':>7} {'n@99%':>7}  top1")
    for r in samp:
        print(f"{r['prompt_index']:>3} {r['temperature']:>5.2f} "
              f"{r['top1_prob']:>9.4f} {r['entropy_nats']:>7.3f} "
              f"{r['effective_vocab']:>10.1f} {r['n_tokens_for_90pct']:>7} "
              f"{r['n_tokens_for_99pct']:>7}  {r['top1_token']!r}")

    for r in kv + att + samp:
        r["model"] = model_id
    return kv, att, samp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seqlens", type=int, nargs="+",
                    default=[128, 256, 512, 1024])
    ap.add_argument("--attn-max-len", type=int, default=512,
                    help="attention weights are n^2 per head per layer; 512 is "
                         "about 150 MB for GPT-2 and is a safe laptop ceiling")
    ap.add_argument("--batch", type=int, default=1,
                    help="u in the handout: concurrent sequences (default 1)")
    ap.add_argument("--device", help="cpu, mps, or cuda (default: best available)")
    ap.add_argument("--dtype", default=None,
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--prompts", nargs="+", default=[
        "The capital of France is",
        "Mike ran a cross country race at Northridge and it was",
        "In conclusion, the most surprising thing about",
    ])
    ap.add_argument("--csv", help="prefix for CSV output")
    ap.add_argument("--skip-attention", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else default_dtype(device)
    args.dtype = str(dtype).replace("torch.", "")

    prov = {"device": device_label(device), "device_type": device.type,
            "dtype": args.dtype, "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform()}
    try:
        import transformers
        prov["transformers"] = transformers.__version__
    except Exception:
        pass
    # PROVENANCE. Paste this line into your report verbatim.
    print("PROVENANCE " + json.dumps(prov))
    if device.type == "cpu":
        print("  Running on CPU. This is expected and supported. The default "
              "run takes roughly two to four minutes.")

    allkv, allatt, allsamp = [], [], []
    for mid in args.models:
        a, b, c = run_one(mid, args, device, dtype)
        allkv += a
        allatt += b
        allsamp += c

    if args.csv:
        for name, rows in (("kv", allkv), ("attention", allatt),
                           ("sampling", allsamp)):
            if not rows:
                continue
            keys = sorted({k for r in rows for k in r})
            path = f"{args.csv}_{name}.csv"
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {path}")

    print("""
Where this output goes (handout section 3.7 has the full map):
  part A  -> Task 1, deliverables T1.1-T1.4 (report section 3)
             Is the ratio exactly 1.000 for both models? If it is, say why the
             formula is exact rather than approximate (T1.1, T1.2). The cache
             is not the largest tensor in the table: what is, how does it
             scale, and under what condition would that change (T1.3, T1.4)?
  part B  -> Task 2, deliverables T2.1-T2.4 (report section 4)
             Does attention get MORE or LESS concentrated as n grows, and is
             the answer the same at layer 0 and at the last layer (T2.3)?
  part C  -> Task 3, deliverables T3.1-T3.4 (report section 5); add your own
             top-k or top-p sweep, which this script does not do.
  no part -> Task 4, deliverables T4.1-T4.8 (report section 6): extrapolate to
             your assigned production model and context length, by hand.
""")


if __name__ == "__main__":
    sys.exit(main())
