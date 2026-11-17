"""32-bit modular addition in the *graph* regime: each ONNX node is a separate kernel / memory pass,
which is how NPU compilers (Hailo, etc.) and DirectML run unfused graphs.

Workload: R dependent additions (t = a + b; a = b; b = t) on N values. Variants:
  native     uint32 Add                                                     (1 node per add)
  ripple     bit-packed [32, M] uint32, split into 32 bit rows, ripple carry (~5 nodes per bit, 32 sequential stages)
  kogge      bit-packed [32, M], Kogge-Stone along the bit axis             (5 prefix levels, whole-tensor ops)
  brent      bit-packed [32, M], Brent-Kung with row masks                  (9 levels, masks instead of fewer gates)
  relu_ks    float 0/1 bits [N, 32], Kogge-Stone with ReLU gates + MatMul shifts (what a Hailo-style NPU runs)
Every result is checked exactly against numpy.

usage: python add_bench.py [--dml] [--n 1048576] [--secs 4]
"""
import argparse, time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from gen_sha256d import Graph, BitPacked, Relu

R = 16
M32 = 0xFFFFFFFF


def fib_ref(a, b):
    a, b = a.astype(np.uint32).copy(), b.astype(np.uint32).copy()
    for _ in range(R):
        a, b = b, (a + b).astype(np.uint32)
    return b


# ---------------------------------------------------------------- bit-packed helpers (ripple + brent)
def ripple_add(g, xs, ys):
    """xs, ys: lists of 32 row tensors [1, M]; returns 32 rows"""
    X = lambda a, b: g.op("BitwiseXor", [a, b]); A = lambda a, b: g.op("BitwiseAnd", [a, b]); O = lambda a, b: g.op("BitwiseOr", [a, b])
    out, c = [], None
    for k in range(32):
        p = X(xs[k], ys[k])
        if c is None:
            out.append(p); c = A(xs[k], ys[k])
        else:
            out.append(X(p, c))
            if k < 31: c = O(A(xs[k], ys[k]), A(c, p))
    return out


def brent_add(B, x, y):
    """Brent-Kung on whole [32, M] tensors: each level updates only a subset of rows, selected by masks"""
    G, P = B.A_(x, y), B.X(x, y)
    P0 = P
    for d in (1, 2, 4, 8, 16):                                    # up-sweep
        rows = set(range(2 * d - 1, 32, 2 * d))
        sel = B.mask(("bk_up", d), [k in rows for k in range(32)])
        keep = B.mask(("bk_upn", d), [k not in rows for k in range(32)])
        G = B.O(G, B.A_(B.A_(P, B.shl(G, d)), sel))
        P = B.A_(P, B.O(B.shl(P, d), keep))
    for d in (8, 4, 2, 1):                                        # down-sweep
        rows = set(range(3 * d - 1, 32, 2 * d))
        sel = B.mask(("bk_dn", d), [k in rows for k in range(32)])
        G = B.O(G, B.A_(B.A_(P, B.shl(G, d)), sel))
    return B.X(P0, B.shl(G, 1))


def build(kind):
    g = Graph()
    if kind == "native":
        ins = [helper.make_tensor_value_info(n, TensorProto.UINT32, ["N"]) for n in ("a", "b")]
        a, b = "a", "b"
        for _ in range(R): a, b = b, g.op("Add", [a, b])
        out_t = TensorProto.UINT32; shape = ["N"]
    elif kind in ("ripple", "kogge", "brent"):
        ins = [helper.make_tensor_value_info(n, TensorProto.UINT32, [32, "M"]) for n in ("a", "b")]
        B = BitPacked(g)
        if kind == "ripple":
            split = g.const(np.ones(32, dtype=np.int64), ("split32",))
            rows = {}
            for n in ("a", "b"):
                outs = [f"{n}_row{k}" for k in range(32)]
                g.nodes.append(helper.make_node("Split", [n, split], outs, axis=0)); rows[n] = outs
            a, b = rows["a"], rows["b"]
            for _ in range(R): a, b = b, ripple_add(g, a, b)
            b = g.op("Concat", b, axis=0)
        else:
            a, b = "a", "b"
            for _ in range(R): a, b = b, (B.ks_add(a, b) if kind == "kogge" else brent_add(B, a, b))
        out_t = TensorProto.UINT32; shape = [32, "M"]
    else:  # relu_ks
        ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["N", 32]) for n in ("a", "b")]
        Rl = Relu(g)
        a, b = "a", "b"
        for _ in range(R): a, b = b, Rl.ks_add(a, b)
        out_t = TensorProto.FLOAT; shape = ["N", 32]
    g.nodes.append(helper.make_node("Identity", [b], ["out"]))
    graph = helper.make_graph(g.nodes, f"add_{kind}", ins, [helper.make_tensor_value_info("out", out_t, shape)], g.inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m)
    return m


# ---------------------------------------------------------------- packing
def pack(v):      # [N] -> [32, N/32]
    v = v.astype(np.uint64).reshape(-1, 32)
    bits = (v[None] >> np.arange(32, dtype=np.uint64)[:, None, None]) & 1
    return (bits << np.arange(32, dtype=np.uint64)).sum(-1).astype(np.uint32)


def unpack(t):    # [32, M] -> [N]
    t = t.astype(np.uint64)
    lanes = (t[:, :, None] >> np.arange(32, dtype=np.uint64)) & 1          # [32 k, M, 32 j]
    return (lanes << np.arange(32, dtype=np.uint64)[:, None, None]).sum(0).reshape(-1).astype(np.uint32)


def feeds(kind, a, b):
    if kind == "native": return {"a": a, "b": b}
    if kind == "relu_ks":
        tb = lambda v: ((v.astype(np.uint64)[:, None] >> np.arange(32, dtype=np.uint64)) & 1).astype(np.float32)
        return {"a": tb(a), "b": tb(b)}
    return {"a": pack(a), "b": pack(b)}


def decode(kind, out):
    if kind == "native": return out
    if kind == "relu_ks": return (np.rint(out).astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1).astype(np.uint32)
    return unpack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dml", action="store_true"); ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--secs", type=float, default=4); ap.add_argument("kinds", nargs="*")
    args = ap.parse_args()
    prov = "DmlExecutionProvider" if args.dml else "CPUExecutionProvider"
    kinds = args.kinds or ["native", "ripple", "kogge", "brent", "relu_ks"]
    rng = np.random.default_rng(1)
    print(f"provider {prov.replace('ExecutionProvider', '')} | N = {args.n} values | R = {R} dependent adds")
    print(f"{'adder':9s} {'nodes':>7s} {'nodes/add':>9s}  {'exact':5s} {'G adds/s':>9s}")
    for kind in kinds:
        m = build(kind)
        so = ort.SessionOptions(); so.log_severity_level = 3
        if args.dml: so.enable_mem_pattern = False
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=[prov])
        n = args.n if kind != "relu_ks" else min(args.n, 1 << 18)       # float bits use 32x the memory
        a = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
        b = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
        f = feeds(kind, a, b)
        ok = np.array_equal(decode(kind, sess.run(None, f)[0]), fib_ref(a, b))
        t0, runs = time.perf_counter(), 0
        while time.perf_counter() - t0 < args.secs:
            sess.run(None, f); runs += 1
        rate = n * R * runs / (time.perf_counter() - t0)
        nodes = len(m.graph.node)
        print(f"{kind:9s} {nodes:7d} {nodes / R:9.0f}  {'OK' if ok else 'FAIL':5s} {rate / 1e9:9.3f}" + (f"   (N={n})" if n != args.n else ""))


if __name__ == "__main__":
    main()
