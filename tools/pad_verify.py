#!/usr/bin/env python3
"""pad_verify.py — prove a pad_tp.py output is exactly the source plus zeros.

    pad_verify.py SRC PADDED

Re-derives the head/row maps from PADDED/config.json `_tp_padding`, then for EVERY tensor in the
padded index checks, streaming one tensor at a time:
  untouched tensors  byte-identical to the source
  padded tensors     every real row/column block equals the source block it came from, and every
                     other element is exactly zero
Exit 0 only if all of it holds. This is the correctness gate for checkpoints no reference forward
exists for (Qwen3.8-Flash-Next).
"""
import json, os, re, struct, sys
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("padtp", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pad_tp.py"))
padtp = importlib.util.module_from_spec(spec); spec.loader.exec_module(padtp)

def load(d):
    cfg_all = json.load(open(os.path.join(d, "config.json")))
    ip = os.path.join(d, "model.safetensors.index.json")
    wm = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None
    files = sorted(set(wm.values())) if wm else ["model.safetensors"]
    hdrs = {}
    for fn in files:
        h, data = padtp.st_read(os.path.join(d, fn)); h.pop("__metadata__", None)
        for k in h: hdrs.setdefault(k, (h, data, fn))      # first by name order = what the engine reads
    return cfg_all, hdrs, wm

def main():
    src, dst = sys.argv[1], sys.argv[2]
    cfg_all, S, _ = load(src); cfgp_all, P, wmp = load(dst)
    tp = cfgp_all["_tp_padding"]; f, t = tp["from"], tp["to"]
    cfg = cfg_all.get("text_config", cfg_all)
    n_head, n_kv, hd = f["heads"], f["kv_heads"], cfg.get("head_dim") or cfg["hidden_size"] // f["heads"]
    qmap = padtp.head_map(n_head, n_kv, t["heads"], t["kv_heads"]); kmap = list(range(n_kv)) + [-1] * (t["kv_heads"] - n_kv)
    gk, gv, gk2, gv2 = f["gdn_k"], f["gdn_v"], t["gdn_k"], t["gdn_v"]
    gkd, gvd = cfg.get("linear_key_head_dim"), cfg.get("linear_value_head_dim")
    bad = 0; n_pad = 0; n_same = 0; zero_bytes = 0

    def blocks_rows(new, old, idx, blk):
        """new rows in blocks of blk follow idx into old; -1 blocks must be zero."""
        ok = new.shape[0] == len(idx) * blk
        for i, o in enumerate(idx):
            seg = new[i*blk:(i+1)*blk]
            ok &= bool(np.array_equal(seg, old[o*blk:(o+1)*blk])) if o >= 0 else not seg.any()
        return ok
    def blocks_cols(new, old, idx, blk):
        ok = new.shape[1] == len(idx) * blk
        for i, o in enumerate(idx):
            seg = new[:, i*blk:(i+1)*blk]
            ok &= bool(np.array_equal(seg, old[:, o*blk:(o+1)*blk])) if o >= 0 else not seg.any()
        return ok
    def segs_rows(new, old, sizes, news):
        o = n = 0; ok = new.shape[0] == sum(news)
        for sz, s2 in zip(sizes, news):
            ok &= bool(np.array_equal(new[n:n+sz], old[o:o+sz])) and not new[n+sz:n+s2].any(); o += sz; n += s2
        return ok and o == old.shape[0]
    def pad_rows_ok(new, old, n2): return new.shape[0] == n2 and np.array_equal(new[:old.shape[0]], old) and not new[old.shape[0]:].any()
    def pad_cols_ok(new, old, n2): return new.shape[1] == n2 and np.array_equal(new[:, :old.shape[1]], old) and not new[:, old.shape[1]:].any()

    for name in sorted(S):
        if name not in P: print("MISSING in padded:", name); bad += 1; continue
        hs, ds, _ = S[name]; hp, dp, fnp = P[name]
        a = padtp.st_tensor(hs, ds, name); b = padtp.st_tensor(hp, dp, name)
        if wmp and wmp.get(name) != fnp: print("INDEX/ORDER mismatch:", name, wmp.get(name), fnp); bad += 1
        ok = None
        if re.search(r"self_attn\.q_proj\.(weight|bias)$", name):   ok = blocks_rows(b, a, qmap, a.shape[0] // n_head)
        elif re.search(r"self_attn\.[kv]_proj\.(weight|bias)$", name): ok = blocks_rows(b, a, kmap, hd)
        elif re.search(r"self_attn\.o_proj\.weight$", name):          ok = blocks_cols(b, a, qmap, hd)
        elif f["ff"] and f["ff"] != t["ff"] and re.search(r"mlp\.(gate|up)_proj\.weight$", name): ok = pad_rows_ok(b, a, t["ff"])
        elif f["ff"] and f["ff"] != t["ff"] and re.search(r"mlp\.down_proj\.weight$", name):      ok = pad_cols_ok(b, a, t["ff"])
        elif gk and (gk2 != gk or gv2 != gv) and re.search(r"linear_attn\.", name):
            qs, qn = [gk*gkd, gk*gkd, gv*gvd], [gk2*gkd, gk2*gkd, gv2*gvd]
            if re.search(r"in_proj_qkvz\.weight$", name):  ok = segs_rows(b, a, qs + [gv*gvd], qn + [gv2*gvd])
            elif re.search(r"in_proj_qkv\.weight$", name): ok = segs_rows(b, a, qs, qn)
            elif re.search(r"in_proj_z\.weight$", name):   ok = pad_rows_ok(b, a, gv2 * gvd)
            elif re.search(r"in_proj_ba\.weight$", name):  ok = segs_rows(b, a, [gv, gv], [gv2, gv2])
            elif re.search(r"in_proj_[ab]\.weight$", name): ok = pad_rows_ok(b, a, gv2)
            elif re.search(r"conv1d\.(weight|bias)$", name): ok = segs_rows(b, a, qs, qn)
            elif re.search(r"(A_log|dt_bias)$", name):     ok = pad_rows_ok(b, a, gv2)
            elif re.search(r"out_proj\.weight$", name):    ok = pad_cols_ok(b, a, gv2 * gvd)
        if ok is None:
            same = a.shape == b.shape and np.array_equal(a, b); n_same += 1
            if not same: print("CHANGED but should be untouched:", name, a.shape, b.shape); bad += 1
        else:
            n_pad += 1; zero_bytes += b.nbytes - a.nbytes
            if not ok: print("PAD MISMATCH:", name, a.shape, "->", b.shape); bad += 1
    extra = set(P) - set(S)
    if extra: print("EXTRA tensors in padded:", sorted(extra)[:5]); bad += 1
    print(f"verified: {n_same} untouched identical, {n_pad} padded exact (+{zero_bytes/1e9:.2f} GB of zeros), {bad} problems")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
