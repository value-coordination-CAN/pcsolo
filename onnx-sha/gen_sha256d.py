"""Generate SHA-256d mining graphs as ONNX models.

Two arithmetic backends share one SHA-256 description:

  u32  - native ONNX integer ops (uint32 Add / BitShift / BitwiseAnd|Or|Xor|Not). Reference + baseline.
  relu - a "neural network": only MatMul, Add, Mul, Relu on 0/1 float tensors.
         * a 32-bit word is a [.., 32] bit vector (index 0 = LSB)
         * rotations / shifts / byte-swap are 32x32 permutation matrices (MatMul)
         * Sigma functions: one MatMul with the summed rotation matrix, then parity of {0..3}
           via s - 2*Relu(s-1) + 2*Relu(s-2)
         * Ch(e,f,g) = Relu(e+f-1) + Relu(g-e);  Maj = Relu(s-1) - Relu(s-2), s = a+b+c
         * multi-operand addition: carry-save adders (3->2), then a Kogge-Stone carry-lookahead adder
           (5 prefix levels), so depth per add is ~log2(32) instead of a 32-step ripple.
         This op set (dense/permutation matmuls + ReLU) is what NPU compilers like Hailo's accept.

Graph inputs (per mining job the first two are constant; the nonce batch varies):
  u32 : mid [8] uint32 (midstate after header block 1), blk [3] uint32 (header words 16..18), nonce [N] uint32
  relu: mid_bits [8,32], blk_bits [3,32], nonce_bits [N,32] float32 (bit i of the nonce value, LSB first)
Output:
  u32 : hash [N,8] uint32  - final SHA-256d state words
  relu: hash_bits [N,256]  - the same 8 words as bits
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

M32 = 0xFFFFFFFF
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

S0_ROT, S1_ROT = (2, 13, 22), (6, 11, 25)       # big Sigma (rounds)
s0_ROT, s1_ROT = ((7, 18), 3), ((17, 19), 10)   # small sigma (message schedule): rotations, shift


def rotr_i(x, n): return ((x >> n) | (x << (32 - n))) & M32


def sigma_i(x, rots, shr):
    v = 0
    for r in rots: v ^= rotr_i(x, r)
    if shr is not None: v ^= x >> shr
    return v


class Graph:
    def __init__(self):
        self.nodes, self.inits, self.n, self.consts = [], [], 0, {}

    def op(self, typ, ins, **attrs):
        self.n += 1
        out = f"{typ.lower()}_{self.n}"
        self.nodes.append(helper.make_node(typ, ins, [out], **attrs))
        return out

    def const(self, arr, key):
        if key not in self.consts:
            name = f"c_{len(self.consts)}"
            self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
            self.consts[key] = name
        return self.consts[key]


# ---------------------------------------------------------------- backends
class U32:
    """Words are uint32 tensors (shape [] or [N]) or Python ints (folded at build time)."""
    def __init__(self, g): self.g = g

    def c(self, v): return self.g.const(np.array(v & M32, dtype=np.uint32), ("u32", v & M32))
    def t(self, v): return self.c(v) if isinstance(v, int) else v

    def rotr(self, x, n):
        g = self.g
        return g.op("BitwiseOr", [g.op("BitShift", [x, self.c(n)], direction="RIGHT"),
                                  g.op("BitShift", [x, self.c(32 - n)], direction="LEFT")])

    def sigma(self, x, rots, shr):
        if isinstance(x, int): return sigma_i(x, rots, shr)
        parts = [self.rotr(x, r) for r in rots]
        if shr is not None: parts.append(self.g.op("BitShift", [x, self.c(shr)], direction="RIGHT"))
        v = parts[0]
        for p in parts[1:]: v = self.g.op("BitwiseXor", [v, p])
        return v

    def ch(self, e, f, h):
        if all(isinstance(v, int) for v in (e, f, h)): return ((e & f) ^ (~e & h)) & M32
        g = self.g
        return g.op("BitwiseXor", [g.op("BitwiseAnd", [self.t(e), self.t(f)]),
                                   g.op("BitwiseAnd", [g.op("BitwiseNot", [self.t(e)]), self.t(h)])])

    def maj(self, a, b, c):
        if all(isinstance(v, int) for v in (a, b, c)): return (a & b) ^ (a & c) ^ (b & c)
        g = self.g; a, b, c = self.t(a), self.t(b), self.t(c)
        return g.op("BitwiseOr", [g.op("BitwiseAnd", [a, b]), g.op("BitwiseAnd", [c, g.op("BitwiseOr", [a, b])])])

    def add(self, terms):
        k = sum(v for v in terms if isinstance(v, int)) & M32
        ts = [v for v in terms if not isinstance(v, int)]
        if not ts: return k
        v = ts[0]
        for x in ts[1:]: v = self.g.op("Add", [v, x])
        if k: v = self.g.op("Add", [v, self.c(k)])
        return v


class Relu:
    """Words are float32 bit tensors (shape [32] or [N,32], LSB first) or Python ints."""
    def __init__(self, g): self.g = g

    def s(self, v): return self.g.const(np.array(v, dtype=np.float32), ("f", float(v)))
    def bits(self, v):
        v &= M32
        return self.g.const(np.array([(v >> i) & 1 for i in range(32)], dtype=np.float32), ("bits", v))
    def t(self, v): return self.bits(v) if isinstance(v, int) else v

    def mat(self, key, fn):
        """32x32 matrix M with y = x @ M; fn(i) -> list of source bits j feeding output bit i."""
        def build():
            m = np.zeros((32, 32), dtype=np.float32)
            for i in range(32):
                for j in fn(i): m[j, i] += 1
            return m
        if key not in self.g.consts: return self.g.const(build(), key)
        return self.g.consts[key]

    def mm(self, x, key, fn): return self.g.op("MatMul", [x, self.mat(key, fn)])
    def add2(self, a, b): return self.g.op("Add", [a, b])
    def sub(self, a, b): return self.g.op("Sub", [a, b])
    def relu_off(self, x, off): return self.g.op("Relu", [self.g.op("Add", [x, self.s(-off)])])  # Relu(x - off)

    def parity3(self, s):                         # s in {0,1,2,3} -> s mod 2
        d = self.sub(self.relu_off(s, 1), self.relu_off(s, 2))
        return self.sub(s, self.g.op("Mul", [d, self.s(2)]))

    def carry3(self, s):                          # s in {0..3} -> s >= 2
        return self.sub(self.relu_off(s, 1), self.relu_off(s, 2))

    def sigma(self, x, rots, shr):
        if isinstance(x, int): return sigma_i(x, rots, shr)
        def src(i):
            out = [(i + r) % 32 for r in rots]
            if shr is not None and i + shr < 32: out.append(i + shr)
            return out
        return self.parity3(self.mm(x, ("sig", rots, shr), src))

    def ch(self, e, f, h):
        if all(isinstance(v, int) for v in (e, f, h)): return ((e & f) ^ (~e & h)) & M32
        e, f, h = self.t(e), self.t(f), self.t(h)
        return self.add2(self.relu_off(self.add2(e, f), 1), self.g.op("Relu", [self.sub(h, e)]))

    def maj(self, a, b, c):
        if all(isinstance(v, int) for v in (a, b, c)): return (a & b) ^ (a & c) ^ (b & c)
        return self.carry3(self.add2(self.add2(self.t(a), self.t(b)), self.t(c)))

    def shl(self, x, d): return self.mm(x, ("shl", d), lambda i: [i - d] if i >= d else [])

    def csa(self, x, y, z):                       # 3 words -> (sum, carry<<1)
        s = self.add2(self.add2(x, y), z)
        c = self.carry3(s)
        return self.sub(s, self.g.op("Mul", [c, self.s(2)])), self.shl(c, 1)

    def ks_add(self, x, y):                       # Kogge-Stone adder mod 2^32
        s = self.add2(x, y)
        gen = self.relu_off(s, 1)                                   # x AND y
        prop = self.sub(s, self.g.op("Mul", [gen, self.s(2)]))      # x XOR y
        G, P = gen, prop
        for d in (1, 2, 4, 8, 16):
            Gs = self.shl(G, d)
            newG = self.add2(G, self.relu_off(self.add2(P, Gs), 1))  # G | (P & G_lo); terms are exclusive
            if d != 16: P = self.relu_off(self.add2(P, self.shl(P, d)), 1)
            G = newG
        cin = self.shl(G, 1)
        t = self.add2(prop, cin)
        return self.sub(t, self.g.op("Mul", [self.relu_off(t, 1), self.s(2)]))  # prop XOR cin

    def add(self, terms):
        k = sum(v for v in terms if isinstance(v, int)) & M32
        ts = [v for v in terms if not isinstance(v, int)]
        if not ts: return k
        if k: ts.append(self.bits(k))
        while len(ts) > 2:                        # Wallace-style carry-save reduction
            nxt = []
            while len(ts) >= 3:
                a, b = self.csa(ts.pop(0), ts.pop(0), ts.pop(0)); nxt += [a, b]
            ts = nxt + ts
        return ts[0] if len(ts) == 1 else self.ks_add(ts[0], ts[1])


class BitPacked:
    """Bitsliced + bit-packed: a word is a uint32 tensor [32, M] - row k holds bit k of the word,
    each uint32 element packs 32 different nonces (one per bit lane). Constant words are [32, 1]
    rows of 0 / 0xFFFFFFFF that broadcast. Only Gather (bit permutation) and Bitwise And/Or/Xor."""
    def __init__(self, g): self.g = g

    def bits(self, v):
        v &= M32
        arr = np.array([[0xFFFFFFFF if (v >> k) & 1 else 0] for k in range(32)], dtype=np.uint32)
        return self.g.const(arr, ("bp", v))
    def t(self, v): return self.bits(v) if isinstance(v, int) else v

    def idx(self, key, lst): return self.g.const(np.array(lst, dtype=np.int64), ("idx",) + key)
    def mask(self, key, lst):
        return self.g.const(np.array([[0xFFFFFFFF if m else 0] for m in lst], dtype=np.uint32), ("mask",) + key)
    def X(self, a, b): return self.g.op("BitwiseXor", [a, b])
    def A_(self, a, b): return self.g.op("BitwiseAnd", [a, b])
    def O(self, a, b): return self.g.op("BitwiseOr", [a, b])

    def rotr(self, x, n):
        return self.g.op("Gather", [x, self.idx(("rotr", n), [(k + n) % 32 for k in range(32)])], axis=0)
    def shr(self, x, n):          # y[k] = x[k+n], zero-filled at the top
        y = self.g.op("Gather", [x, self.idx(("shr", n), [min(k + n, 31) for k in range(32)])], axis=0)
        return self.A_(y, self.mask(("shr", n), [k + n < 32 for k in range(32)]))
    def shl(self, x, n):          # y[k] = x[k-n], zero-filled at the bottom (carry direction)
        y = self.g.op("Gather", [x, self.idx(("shl", n), [max(k - n, 0) for k in range(32)])], axis=0)
        return self.A_(y, self.mask(("shl", n), [k >= n for k in range(32)]))

    def sigma(self, x, rots, shr):
        if isinstance(x, int): return sigma_i(x, rots, shr)
        parts = [self.rotr(x, r) for r in rots] + ([self.shr(x, shr)] if shr is not None else [])
        v = parts[0]
        for p in parts[1:]: v = self.X(v, p)
        return v

    def ch(self, e, f, h):
        if all(isinstance(v, int) for v in (e, f, h)): return ((e & f) ^ (~e & h)) & M32
        e, f, h = self.t(e), self.t(f), self.t(h)
        return self.X(h, self.A_(e, self.X(f, h)))

    def maj(self, a, b, c):
        if all(isinstance(v, int) for v in (a, b, c)): return (a & b) ^ (a & c) ^ (b & c)
        a, b, c = self.t(a), self.t(b), self.t(c)
        return self.O(self.A_(a, b), self.A_(c, self.O(a, b)))

    def csa(self, x, y, z):       # 3 words -> (sum, carry<<1), no carry propagation
        xy = self.X(x, y)
        return self.X(xy, z), self.shl(self.O(self.A_(x, y), self.A_(z, xy)), 1)

    def ks_add(self, x, y):       # Kogge-Stone along the bit axis: 5 prefix levels
        G, P = self.A_(x, y), self.X(x, y)
        P0 = P
        for d in (1, 2, 4, 8, 16):
            newG = self.O(G, self.A_(P, self.shl(G, d)))
            if d != 16: P = self.A_(P, self.shl(P, d))
            G = newG
        return self.X(P0, self.shl(G, 1))

    def add(self, terms):
        k = sum(v for v in terms if isinstance(v, int)) & M32
        ts = [v for v in terms if not isinstance(v, int)]
        if not ts: return k
        if k: ts.append(self.bits(k))
        while len(ts) > 2:
            nxt = []
            while len(ts) >= 3:
                a, b = self.csa(ts.pop(0), ts.pop(0), ts.pop(0)); nxt += [a, b]
            ts = nxt + ts
        return ts[0] if len(ts) == 1 else self.ks_add(ts[0], ts[1])


# ---------------------------------------------------------------- SHA-256
def compress(A, state, w):
    w = list(w)
    for i in range(16, 64):
        w.append(A.add([A.sigma(w[i - 2], *s1_ROT), w[i - 7], A.sigma(w[i - 15], *s0_ROT), w[i - 16]]))
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = A.add([h, A.sigma(e, S1_ROT, None), A.ch(e, f, g), K[i], w[i]])
        t2 = [A.sigma(a, S0_ROT, None), A.maj(a, b, c)]
        h, g, f = g, f, e
        e = A.add([d, t1])
        d, c, b = c, b, a
        a = A.add([t1] + t2)
    return [A.add([state[i], v]) for i, v in enumerate([a, b, c, d, e, f, g, h])]


def build(kind):
    g = Graph()
    if kind == "u32":
        A = U32(g)
        idx = lambda i: g.const(np.array(i, dtype=np.int64), ("i64", i))
        mid = [g.op("Gather", ["mid", idx(i)], axis=0) for i in range(8)]
        blk = [g.op("Gather", ["blk", idx(i)], axis=0) for i in range(3)]
        x = "nonce"   # byte-swap: header stores the nonce little-endian, SHA reads big-endian words
        w3 = g.op("BitwiseOr", [
            g.op("BitwiseOr", [g.op("BitShift", [x, A.c(24)], direction="RIGHT"),
                               g.op("BitwiseAnd", [g.op("BitShift", [x, A.c(8)], direction="RIGHT"), A.c(0xff00)])]),
            g.op("BitwiseOr", [g.op("BitwiseAnd", [g.op("BitShift", [x, A.c(8)], direction="LEFT"), A.c(0xff0000)]),
                               g.op("BitShift", [x, A.c(24)], direction="LEFT")])])
        inputs = [helper.make_tensor_value_info("mid", TensorProto.UINT32, [8]),
                  helper.make_tensor_value_info("blk", TensorProto.UINT32, [3]),
                  helper.make_tensor_value_info("nonce", TensorProto.UINT32, ["N"])]
    elif kind == "bits":
        A = BitPacked(g)
        idx = lambda i: g.const(np.array(i, dtype=np.int64), ("i64", i))
        mid = [g.op("Gather", ["mid_bs", idx(i)], axis=0) for i in range(8)]   # each [32, 1]
        blk = [g.op("Gather", ["blk_bs", idx(i)], axis=0) for i in range(3)]
        # byte-swap = bit permutation along the bit axis
        w3 = g.op("Gather", ["nonce_bs", A.idx(("bswap",), [(3 - k // 8) * 8 + k % 8 for k in range(32)])], axis=0)
        inputs = [helper.make_tensor_value_info("mid_bs", TensorProto.UINT32, [8, 32, 1]),
                  helper.make_tensor_value_info("blk_bs", TensorProto.UINT32, [3, 32, 1]),
                  helper.make_tensor_value_info("nonce_bs", TensorProto.UINT32, [32, "M"])]
    else:
        A = Relu(g)
        idx = lambda i: g.const(np.array(i, dtype=np.int64), ("i64", i))
        mid = [g.op("Gather", ["mid_bits", idx(i)], axis=0) for i in range(8)]
        blk = [g.op("Gather", ["blk_bits", idx(i)], axis=0) for i in range(3)]
        # byte-swap is just a bit permutation: output bit i (byte i//8) <- input byte 3 - i//8
        w3 = A.mm("nonce_bits", ("bswap",), lambda i: [(3 - i // 8) * 8 + i % 8])
        inputs = [helper.make_tensor_value_info("mid_bits", TensorProto.FLOAT, [8, 32]),
                  helper.make_tensor_value_info("blk_bits", TensorProto.FLOAT, [3, 32]),
                  helper.make_tensor_value_info("nonce_bits", TensorProto.FLOAT, ["N", 32])]

    h1 = compress(A, mid, blk + [w3, 0x80000000] + [0] * 10 + [640])
    h2 = compress(A, IV, h1 + [0x80000000] + [0] * 6 + [256])

    if kind == "u32":
        cols = [g.op("Unsqueeze", [A.t(v), g.const(np.array([1], dtype=np.int64), ("ax1",))]) for v in h2]
        out = g.op("Concat", cols, axis=1)
        outputs = [helper.make_tensor_value_info("hash", TensorProto.UINT32, ["N", 8])]
        g.nodes.append(helper.make_node("Identity", [out], ["hash"]))
    elif kind == "bits":
        cols = [g.op("Unsqueeze", [A.t(v), g.const(np.array([0], dtype=np.int64), ("ax0",))]) for v in h2]
        out = g.op("Concat", cols, axis=0)
        outputs = [helper.make_tensor_value_info("hash_bs", TensorProto.UINT32, [8, 32, "M"])]
        g.nodes.append(helper.make_node("Identity", [out], ["hash_bs"]))
    else:
        out = g.op("Concat", [A.t(v) for v in h2], axis=-1)
        outputs = [helper.make_tensor_value_info("hash_bits", TensorProto.FLOAT, ["N", 256])]
        g.nodes.append(helper.make_node("Identity", [out], ["hash_bits"]))

    graph = helper.make_graph(g.nodes, f"sha256d_{kind}", inputs, outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="pcsolo-onnx-sha")
    model.ir_version = 9
    return model


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["u32", "relu", "bits"]
    for kind in kinds:
        m = build(kind)
        onnx.checker.check_model(m)
        path = f"sha256d_{kind}.onnx"
        onnx.save(m, path)
        ops = {}
        for n in m.graph.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print(f"{path}: {len(m.graph.node)} nodes, {len(m.graph.initializer)} constants")
        print("   ", ", ".join(f"{k} {v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
