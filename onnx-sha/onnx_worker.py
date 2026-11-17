"""Run a SHA-256d ONNX graph continuously and log its hash rate every 2 s (for contention tests).

usage: python onnx_worker.py <u32|relu|bits> <cpu|dml> <seconds> [batch]
Output lines: '<unix time> <kH/s>'
"""
import sys, time
import numpy as np
import test_bench as tb

kind, prov, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1 << 20
sess, _ = tb.session(kind, "DmlExecutionProvider" if prov == "dml" else "CPUExecutionProvider")
nonces = np.arange(n, dtype=np.uint32)
f = tb.feeds(kind, nonces)
sess.run(None, f)  # warm-up
print(f"{time.time():.1f} START", flush=True)
end = time.time() + secs
t0, done = time.time(), 0
while time.time() < end:
    sess.run(None, f); done += n
    t = time.time()
    if t - t0 >= 2:
        print(f"{t:.1f} {done / (t - t0) / 1e3:.1f}", flush=True)
        t0, done = t, 0
