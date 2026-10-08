#!/usr/bin/env python3
# cardclock.py -- what holds each card's clock under a sustained load, and which card sets the step.
#
# A tensor-parallel step ends when its slowest rank finishes: every rank waits for the others in the
# all-reduce. So the clock that decides decode speed is the LOWEST of the cards', and on a
# workstation card under a sustained load the firmware -- not the engine -- picks that clock: the
# power limit (PPT), a temperature limit, a current limit. The same image then reads several percent
# apart from one boot to the next, depending on which card heats first and how hard its power limit
# bites. This puts a number on that:
#
#   * drives a sustained greedy load at --conc against a live server (or, with --watch, only samples
#     while something else drives it -- real traffic on a deployment), and reports tok/s per --window
#     seconds;
#   * beside it, the engine's own step counters per window (/metrics): step ms, tokens per decode step,
#     draft acceptance, busy share and host gap per step -- so a slow window is attributed to slower
#     steps, to the text (fewer tokens a step), or to the host, before anyone blames a clock. The
#     engine's tok/s counts every request the server ran in the window, the load's only this script's:
#     a gap between the two is someone else's traffic;
#   * reads every AMD card's SMU metrics table (/sys/class/drm/card*/device/gpu_metrics) ten times a
#     second: clocks, temperatures (edge, hotspot, memory, the GFX/SOC/MEM voltage regulators), power,
#     fan, and the throttler bits the firmware itself sets -- READ-ONLY, no root needed;
#   * names the card the step waits for and what holds it, and the setting that addresses that;
#   * --json saves a run; --compare puts two of them side by side (before / after a setting change).
#
#   scripts/cardclock.py [--port 8100] [--conc 8] [--seconds 180] [--gen 1024] [--json run.json]
#   scripts/cardclock.py --watch [--seconds 300] [--json run.json]
#   scripts/cardclock.py --compare before.json after.json
#
# Reading the throttler bits: a bit counts only when it is set under load AND the card's clock is
# lower while it is set. Some firmware keeps a bit set all the time -- on Radeon AI PRO R9700
# (gfx1201) TEMP_HOTSPOT reads set at idle and at 45 C with no effect on the clock -- so a bit that
# the idle baseline also shows is listed and ignored.
#
# Changing a setting this points at (power1_cap, fan target, voltage offset) is a write to the SMU:
# do it ONLY while no process holds /dev/kfd -- stop the server first. On a gfx1201 box such a write
# under load has reset the machine. Check with `sudo find /proc/[0-9]*/fd -lname /dev/kfd`, not
# `lsof /dev/kfd`: lsof on the host does not see a containerised server's handle (the container's
# /dev/kfd is its own node). None of these settings survive a reboot. Undervolts in
# particular must be proven under a SUSTAINED hot load, not a short test: an offset that passes a
# minute-long selftest can wedge a card twenty minutes into real traffic once it runs at 90 C.
#
# Needs Linux amdgpu with a gpu_metrics v1.3 table (RDNA3 / RDNA4 discrete cards); any other layout
# is refused rather than misread.
import argparse, glob, json, os, random, re, statistics as st, struct, sys, threading, time, urllib.request

BITS = {0: "PPT0", 1: "PPT1", 2: "PPT2", 3: "PPT3", 4: "SPL", 5: "FPPT", 6: "SPPT", 7: "SPPT_APU",
        16: "TDC_GFX", 17: "TDC_SOC", 18: "TDC_MEM", 19: "TDC_VDD", 20: "TDC_CVIP", 21: "EDC_CPU",
        22: "EDC_GFX", 23: "APCC", 32: "TEMP_GPU", 33: "TEMP_CORE", 34: "TEMP_MEM", 35: "TEMP_EDGE",
        36: "TEMP_HOTSPOT", 37: "TEMP_SOC", 38: "TEMP_VR_GFX", 39: "TEMP_VR_SOC", 40: "TEMP_VR_MEM0",
        41: "TEMP_VR_MEM1", 42: "TEMP_LIQUID0", 43: "TEMP_LIQUID1", 44: "VRHOT0", 45: "VRHOT1",
        46: "PROCHOT_CPU", 47: "PROCHOT_GFX", 56: "PPM", 57: "FIT"}          # amdgpu_smu.h SMU_THROTTLER_*_BIT
V13 = "<HBB6H4HQQ7H7HIHHHHII4HQHHHHQ"                                      # gpu_metrics_v1_3, 120 bytes
ADVICE = [
    (("PPT", "SPL", "FPPT", "SPPT"), "power limit",
     "raise power1_cap toward power1_cap_max (now {cap} of {cap_max} W), or lower watts per clock with a "
     "negative VDDGFX offset proven under sustained load",
     "echo {cap_max}000000 | sudo tee {hwmon}/power1_cap"),
    (("TEMP", "VRHOT", "PROCHOT"), "temperature limit",
     "more cooling: lower the fan target (now {fan_target} C) or improve airflow; a cooler card also "
     "draws fewer watts for the same clock",
     "echo 65 | sudo tee {dev}/gpu_od/fan_ctrl/fan_target_temperature && echo c | sudo tee {dev}/gpu_od/fan_ctrl/fan_target_temperature"),
    (("TDC", "EDC"), "current limit", "a negative VDDGFX offset (fewer amps per clock), proven under sustained load",
     "echo 'vo -50' | sudo tee {dev}/pp_od_clk_voltage && echo c | sudo tee {dev}/pp_od_clk_voltage"),
    (("FIT", "APCC", "PPM"), "reliability / platform limit", "nothing to tune from the host", ""),
]


def rd(p, conv=str, default=None):
    try: return conv(open(p).read().strip())
    except (OSError, ValueError): return default


class Card:
    def __init__(self, dev):
        self.dev = dev; self.bdf = os.path.basename(os.path.realpath(dev))
        hw = glob.glob(dev + "/hwmon/hwmon*"); self.hwmon = hw[0] if hw else None

    def read(self):
        b = open(self.dev + "/gpu_metrics", "rb").read()
        size, fr, cr = struct.unpack_from("<HBB", b)
        if (fr, cr) != (1, 3) or size != 120 or len(b) < 120:
            raise SystemExit(f"{self.bdf}: gpu_metrics v{fr}.{cr} ({size} bytes) -- only the v1.3 layout is known here")
        v = struct.unpack_from(V13, b)
        return {"edge": v[3], "hot": v[4], "mem": v[5], "vrgfx": v[6], "vrsoc": v[7], "vrmem": v[8],
                "act": v[9], "W": v[12], "clk": v[15], "fan": v[30], "mv": v[42], "indep": v[45]}

    def check(self):
        r = self.read(); jn = rd(f"{self.hwmon}/temp2_input", int) if self.hwmon else None
        rpm = rd(f"{self.hwmon}/fan1_input", int) if self.hwmon else None
        bad = (jn is not None and abs(r["hot"] - jn // 1000) > 5) or (rpm and abs(r["fan"] - rpm) > 400) or not 300 < r["mv"] < 1600
        if bad: raise SystemExit(f"{self.bdf}: decoded metrics disagree with hwmon (hotspot {r['hot']} vs {jn}, fan {r['fan']} vs {rpm}, "
                                 f"{r['mv']} mV) -- refusing to report from a misread table")

    def settings(self):
        od = rd(self.dev + "/pp_od_clk_voltage") or ""
        m = re.search(r"OD_VDDGFX_OFFSET:\s*\n\s*(-?\d+)mV", od)
        ft = rd(self.dev + "/gpu_od/fan_ctrl/fan_target_temperature") or ""
        ft = re.search(r"FAN_TARGET_TEMPERATURE:\s*\n\s*(\d+)", ft)
        w = lambda f: rd(f"{self.hwmon}/{f}", int, 0) // 1_000_000 if self.hwmon else None
        return {"cap": w("power1_cap"), "cap_max": w("power1_cap_max"), "cap_default": w("power1_cap_default"),
                "vddgfx_offset_mv": int(m.group(1)) if m else None, "fan_target": int(ft.group(1)) if ft else None,
                "dev": self.dev, "hwmon": self.hwmon}


def find_cards():
    out = {}
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        if not re.fullmatch(r"card\d+", dev.split("/")[-2]): continue
        if rd(dev + "/vendor") != "0x1002" or not os.path.exists(dev + "/gpu_metrics"): continue
        c = Card(dev); out.setdefault(c.bdf, c)
    if not out: raise SystemExit("no AMD card with a gpu_metrics table under /sys/class/drm")
    return list(out.values())


class Sampler(threading.Thread):
    def __init__(self, cards, interval):
        super().__init__(daemon=True); self.cards, self.iv, self.rows, self.stop = cards, interval, [], threading.Event()
    def run(self):
        while not self.stop.is_set():
            t = time.time()
            for c in self.cards:
                try: self.rows.append({"t": t, "card": c.bdf, **c.read()})
                except OSError: pass
            self.stop.wait(max(0.0, self.iv - (time.time() - t)))


TOPICS = ["how a B-tree splits and merges nodes", "the history of the printing press", "how TCP congestion control works",
          "the water cycle and its effect on climate", "how a compiler register allocator works", "the economics of container shipping",
          "how vaccines train the immune system", "the design of a key-value store", "how bridges carry load", "the life of a star"]


def post(base, body, timeout=900):
    return json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                                    headers={"content-type": "application/json"}), timeout=timeout))


ENGINE = ("decode_tokens_total", "engine_steps_total", "engine_steps_decode_total", "engine_steps_mixed_total",
          "engine_step_seconds_total", "engine_step_seconds_decode_total", "draft_tokens_total", "draft_accepted_total")


def engine_counters(base):
    """The engine's own step counters (Radiance /metrics, summed over labels); None if the server has none."""
    try: m = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
    except Exception: return None
    out = {}
    for k in ENGINE:
        v = re.findall(rf"^radiance:{k}(?:\{{[^}}]*\}})? ([0-9.e+-]+)$", m, re.M)
        if v: out[k] = sum(float(x) for x in v)
    return out or None


def engine_windows(snaps, t0, t1, w):
    """Per window: what the engine did -- step ms, tokens per decode step (speculation), draft acceptance, the share of
    wall time inside steps (busy) and the host gap per step. Tells a slow window's cause apart: slower steps = the GPUs
    / interconnect; fewer tokens per step = the text (speculation accepts less); a bigger gap = the host."""
    out = []; k = 0
    def at(t):   # last snapshot at or before t
        c = [s for s in snaps if s[0] <= t]; return c[-1] if c else None
    while t0 + (k + 1) * w <= t1 + 0.5:
        a, b = at(t0 + k * w), at(t0 + (k + 1) * w); k += 1
        if not a or not b or b[0] <= a[0]: out.append(None); continue
        d = {x: b[1].get(x, 0) - a[1].get(x, 0) for x in ENGINE}; wall = b[0] - a[0]
        sd, so = max(d["engine_steps_decode_total"], 1), max(d["engine_steps_total"], 1)
        out.append({"tok_s": round(d["decode_tokens_total"] / wall, 1),
                    "dec_ms": round(d["engine_step_seconds_decode_total"] / sd * 1e3, 2),
                    "tok_step": round(d["decode_tokens_total"] / sd, 2),
                    "accept": round(d["draft_accepted_total"] / d["draft_tokens_total"], 3) if d["draft_tokens_total"] else None,
                    "busy": round(d["engine_step_seconds_total"] / wall, 3),
                    "gap_ms": round((wall - d["engine_step_seconds_total"]) / so * 1e3, 2),
                    "steps_s": round(so / wall, 1), "mixed": int(d["engine_steps_mixed_total"])})
    return out


def load(a, base, model):
    done, stop, stall = [], threading.Event(), {}
    def worker(i):
        # --seed: every run sends the same prompts in the same order per stream (the text drives speculation, so a
        # run-to-run A/B wants it fixed); without it, a fresh mix each run
        rnd = random.Random(i * 7919 + (a.seed if a.seed is not None else int(time.time())))
        while not stop.is_set():
            t0 = time.time(); tag, topic = rnd.getrandbits(48), rnd.randrange(len(TOPICS))
            body = {"model": model, "max_tokens": a.gen, "temperature": 0, "messages": [{"role": "user", "content":
                    f"[{tag:x}] Write a long, detailed, multi-section explanation of {TOPICS[topic]}."}]}
            try: n = post(base, body)["usage"]["completion_tokens"]
            except Exception as e: done.append((t0, time.time(), 0, str(e)[:120], i, topic)); time.sleep(1); continue
            done.append((t0, time.time(), n, None, i, topic))
    ws = [threading.Thread(target=lambda: post(base, {"model": model, "max_tokens": 64, "temperature": 0,   # warm-up: the first
                           "messages": [{"role": "user", "content": "Say hello."}]}), daemon=True) for _ in range(a.conc)]  # requests after a boot read slow
    [w.start() for w in ws]; [w.join() for w in ws]
    snaps = []
    def tokens():
        c = engine_counters(base)
        if c: snaps.append((time.time(), c))
        return c.get("decode_tokens_total") if c else None
    last = tokens()                                            # the baseline the first window is differenced against
    t_start = time.time(); end = t_start + a.seconds
    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.conc)]; [t.start() for t in th]
    last_t = time.time()
    while time.time() < end:                                   # stall guard: a wedged rank stops every token
        time.sleep(2); now = tokens()
        if now is None or now != last: last, last_t = now, time.time()
        elif time.time() - last_t > a.stall:
            stall["at"] = time.time(); print(f"STALL: the server produced no token for {a.stall:.0f} s -- a rank may be wedged; "
                                             "check the server log ('a synchronisation has waited') before anything else", flush=True)
            break
    stop.set(); [t.join(timeout=5 if stall else a.gen) for t in th]
    return t_start, min(end, stall.get("at", end)), done, bool(stall), snaps


def windows(t0, t1, done, w):
    out = []; k = 0
    while t0 + (k + 1) * w <= t1 + 0.5:
        a, b = t0 + k * w, t0 + (k + 1) * w; tok = 0.0
        for s, e, n, *_ in done:
            if n and e > a and s < b: tok += n * (min(b, e) - max(a, s)) / max(e - s, 1e-6)
        out.append(round(tok / w, 1)); k += 1
    return out


def pct(xs, q): xs = sorted(xs); return xs[int(q * (len(xs) - 1))] if xs else None


def analyse(cards, rows, t0, t1, idle_rows, settings):
    rep = {}
    for c in cards:
        rs = [r for r in rows if r["card"] == c.bdf and t0 <= r["t"] <= t1 and r["act"] >= 50]
        if len(rs) < 20: continue
        idle = [r for r in idle_rows if r["card"] == c.bdf]
        flags = {}
        for bit, name in BITS.items():
            on = [r for r in rs if r["indep"] >> bit & 1]
            if not on: continue
            off = [r for r in rs if not r["indep"] >> bit & 1]
            idle_on = sum(1 for r in idle if r["indep"] >> bit & 1) / len(idle) if idle else 0.0
            c_on, c_off = st.median(r["clk"] for r in on), (st.median(r["clk"] for r in off) if off else None)
            share = len(on) / len(rs)
            if idle_on >= 0.2: status = "idle"            # the firmware sets it with no load: says nothing about the load
            elif share > 0.95:                            # never clear under load: no clear samples to compare against --
                cap = settings[c.bdf]["cap"] or 0         # but a power limit set throughout at the cap is the cap
                at_cap = name.startswith(("PPT", "SPL", "FPPT", "SPPT")) and cap and st.median(r["W"] for r in rs) >= 0.95 * cap
                status = "binding" if at_cap else "always"
            elif share >= 0.10 and c_on < c_off * 0.99: status = "binding"
            else: status = "no-effect"
            flags[name] = {"share": round(share, 3), "clk_on": round(c_on), "clk_off": round(c_off) if c_off else None,
                           "idle_share": round(idle_on, 2), "status": status, "binding": status == "binding"}
        rep[c.bdf] = {"samples": len(rs), "clk_p10": pct([r["clk"] for r in rs], .1), "clk_p50": pct([r["clk"] for r in rs], .5),
                      "clk_p90": pct([r["clk"] for r in rs], .9), "W_p50": pct([r["W"] for r in rs], .5), "W_max": max(r["W"] for r in rs),
                      "hot_max": max(r["hot"] for r in rs), "mem_max": max(r["mem"] for r in rs), "vrgfx_max": max(r["vrgfx"] for r in rs),
                      "fan_max": max(r["fan"] for r in rs), "mv_p50": pct([r["mv"] for r in rs], .5), "flags": flags}
    return rep


def advise(bdf, r, s):
    lines = []
    for name, f in sorted(r["flags"].items(), key=lambda x: -x[1]["share"]):
        if not f["binding"]: continue
        for prefixes, what, fix, cmd in ADVICE:
            if name.startswith(prefixes):
                vs = f"{f['clk_on']} MHz while set vs {f['clk_off']} MHz while clear" if f["clk_off"] and f["clk_on"] < f["clk_off"] else \
                     f"set nearly throughout, at {r['W_p50']} W of a {s.get('cap')} W cap"
                lines.append(f"    {name} ({what}) holds it {f['share']:.0%} of the time: {vs} -> "
                             f"{fix.format(**{k: s.get(k) for k in ('cap', 'cap_max', 'fan_target')})}")
                if cmd: lines.append(f"      {cmd.format(cap_max=s.get('cap_max'), hwmon=s.get('hwmon'), dev=s.get('dev'))}")
                break
    return lines


def report(res):
    rep, sets, tps = res["cards"], res["settings"], res["tok_s_windows"]
    if tps: print(f"load: c{res['conc']} gen {res['gen']} for {res['seconds']:.0f} s -> tok/s per {res['window']} s: "
                  f"{' '.join(f'{x:.0f}' for x in tps)}  (median {st.median(tps[1:] or tps):.0f}, first window -- the prefills -- left out)")
    eng = [e for e in res.get("engine_windows") or [] if e]
    if eng:
        f = lambda k: " ".join(f"{e[k]}" if e[k] is not None else "-" for e in res["engine_windows"] if e)
        print(f"engine per window: tok/s {f('tok_s')}\n                   step ms {f('dec_ms')}\n                   tok/step {f('tok_step')}\n"
              f"                   accept {f('accept')}\n                   busy {f('busy')}  host gap ms {f('gap_ms')}")
    if not rep:
        print("no card was busy (>= 50 %) for 20 samples in the window -- nothing to attribute"); return
    print(f"{'card':14s} {'clock p10/p50/p90':>18s} {'W p50/max cap':>15s} {'hot':>4s} {'mem':>4s} {'VR':>4s} {'fan':>5s} {'mV':>5s}  offset  limits that bind (share of loaded samples)")
    for bdf, r in sorted(rep.items()):
        s = sets[bdf]; bind = ", ".join(f"{n} {f['share']:.0%}" for n, f in r["flags"].items() if f["binding"]) or "none"
        print(f"{bdf:14s} {r['clk_p10']:>6}/{r['clk_p50']}/{r['clk_p90']} {r['W_p50']:>5}/{r['W_max']} {s['cap']:>3} "
              f"{r['hot_max']:>4} {r['mem_max']:>4} {r['vrgfx_max']:>4} {r['fan_max']:>5} {r['mv_p50']:>5}  {str(s['vddgfx_offset_mv'] if s['vddgfx_offset_mv'] is not None else '-'):>6}  {bind}")
        ign = [n for n, f in r["flags"].items() if f["status"] == "idle"]
        alw = [n for n, f in r["flags"].items() if f["status"] == "always"]
        if ign: print(f"{'':14s} ignored, also set at idle: {', '.join(ign)}")
        if alw: print(f"{'':14s} set through the whole load, never clear -- cannot tell whether it limits: {', '.join(alw)}")
    if len(rep) >= 2:
        order = sorted(rep.items(), key=lambda x: x[1]["clk_p50"]); (b0, r0), (_, r1) = order[0], order[1]
        gap = 1 - r0["clk_p50"] / r1["clk_p50"]
        if gap >= 0.015:
            print(f"\nthe step waits for {b0}: its median clock {r0['clk_p50']} MHz is {gap:.1%} under the next card's")
            ad = advise(b0, r0, sets[b0])
            for l in ad: print(l)
            if not ad and r0["clk_p90"] <= r0["clk_p50"] * 1.01:
                peers = ", ".join(f"{r['mv_p50']} mV" for b, r in order[1:])
                print(f"    no limit binds and its clock sits flat at ~{r0['clk_p90']} MHz on {r0['mv_p50']} mV (others: {peers}): it is at the\n"
                      f"    top of its voltage/frequency curve. A higher power cap or more cooling will not move it; only a voltage\n"
                      f"    offset would -- and an offset must be proven under a sustained hot load, not a short test.")
        else: print(f"\nno single straggler: the busiest cards' median clocks are within {gap:.1%} of each other")
        for b, r in order[1:]:
            ad = advise(b, r, sets[b])
            if ad: print(f"also on {b}:"); [print(l) for l in ad]
    print("\nwrite a setting only with the server stopped -- nothing may hold /dev/kfd "
          "(sudo find /proc/[0-9]*/fd -lname /dev/kfd); none survive a reboot.")


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    ta, tb = A["tok_s_windows"][1:], B["tok_s_windows"][1:]       # first window = the prefills
    if ta and tb:
        ma, mb = st.median(ta), st.median(tb); iq = lambda x: pct(x, .75) - pct(x, .25)
        spread = max(iq(ta), iq(tb)); d = mb - ma
        print(f"tok/s per window  A median {ma:.0f} (IQR {iq(ta):.0f})   B median {mb:.0f} (IQR {iq(tb):.0f})   B-A {d:+.0f} ({d / ma:+.1%}) -> "
              + ("beyond either run's window spread" if abs(d) > spread else "within the window spread"))
    print(f"{'card':14s} {'clock p10 A->B':>16s} {'clock p50 A->B':>16s} {'W p50 A->B':>12s} {'cap A->B':>10s} {'hot max A->B':>14s}  binding limits A | B")
    for bdf in sorted(set(A["cards"]) | set(B["cards"])):
        a, b = A["cards"].get(bdf), B["cards"].get(bdf)
        if not (a and b): print(f"{bdf:14s} only in {'A' if a else 'B'}"); continue
        bind = lambda r: ",".join(f"{n} {f['share']:.0%}" for n, f in r["flags"].items() if f["binding"]) or "none"
        print(f"{bdf:14s} {a['clk_p10']:>7}->{b['clk_p10']:<7} {a['clk_p50']:>7}->{b['clk_p50']:<7} {a['W_p50']:>5}->{b['W_p50']:<5} "
              f"{A['settings'][bdf]['cap']:>4}->{B['settings'][bdf]['cap']:<4} {a['hot_max']:>6}->{b['hot_max']:<6}  {bind(a)} | {bind(b)}")
    print("one run each is one boot: on these cards boots differ by several percent -- repeat each side >= 3 times, interleaved.")


ap = argparse.ArgumentParser(description="what holds each card's clock under load (read-only)")
ap.add_argument("--port", type=int, default=8100); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--model")
ap.add_argument("--conc", type=int, default=8); ap.add_argument("--gen", type=int, default=1024)
ap.add_argument("--seconds", type=float, default=180); ap.add_argument("--window", type=int, default=20)
ap.add_argument("--interval", type=float, default=0.1); ap.add_argument("--stall", type=float, default=20)
ap.add_argument("--cool", type=int, help="wait (up to 5 min) until every card's hotspot is at or under this before loading")
ap.add_argument("--watch", action="store_true", help="sample only; drive the load yourself")
ap.add_argument("--json"); ap.add_argument("--raw", help="also write every sample (JSON lines)")
ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
ap.add_argument("--seed", type=int, help="fixed prompt seed: the same text every run (speculation makes tok/s text-dependent)")
a = ap.parse_args()
if a.compare: compare(*a.compare); sys.exit(0)
cards = find_cards(); [c.check() for c in cards]
base = f"http://{a.host}:{a.port}"
if a.cool:
    t = time.time()
    while time.time() - t < 300 and max(c.read()["hot"] for c in cards) > a.cool: time.sleep(2)
smp = Sampler(cards, a.interval); smp.start()
time.sleep(5); idle = [r for r in smp.rows if r["act"] < 20]   # idle baseline: which bits the firmware sets with no load
settings = {c.bdf: c.settings() for c in cards}
if a.watch:                                     # no load of our own; the server's step counters if one answers on --port
    snaps = []
    def snap():
        c = engine_counters(base)
        if c: snaps.append((time.time(), c))
    snap(); t0 = time.time()
    while time.time() - t0 < a.seconds: time.sleep(min(2, max(0.0, a.seconds - (time.time() - t0)))); snap()
    t1 = time.time(); tps, stalled, conc, done = [], False, None, []
    eng = engine_windows(snaps, t0, t1, a.window) if snaps else []
else:
    model = a.model or json.load(urllib.request.urlopen(base + "/v1/models", timeout=10))["data"][0]["id"]
    t0, t1, done, stalled, snaps = load(a, base, model); tps = windows(t0, t1, done, a.window); conc = a.conc
    eng = engine_windows(snaps, t0, t1, a.window)
    errs = [d[3] for d in done if d[3]]
    if errs: print(f"{len(errs)} request(s) failed, first: {errs[0]}")
smp.stop.set(); smp.join()
res = {"when": time.strftime("%F %T"), "conc": conc, "gen": a.gen, "seconds": t1 - t0, "window": a.window, "stalled": stalled,
       "seed": a.seed, "t0": t0, "tok_s_windows": tps, "engine_windows": eng,
       "requests": [{"t0": round(d[0], 2), "t1": round(d[1], 2), "tokens": d[2], "err": d[3], "stream": d[4], "topic": d[5]} for d in done],
       "settings": settings, "cards": analyse(cards, smp.rows, t0, t1, idle, settings)}
report(res)
if a.json: json.dump(res, open(a.json, "w"), indent=1)
if a.raw:
    with open(a.raw, "w") as f: [f.write(json.dumps(r) + "\n") for r in smp.rows]
sys.exit(3 if stalled else 0)
