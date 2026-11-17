"""Verify the ONNX SHA-256d graphs against hashlib on the genesis block, then benchmark.

usage: python test_bench.py [u32|relu ...] [--cpu] [--dml] [--sizes 1024,8192] [--secs 10]
"""
import argparse, hashlib, struct, time
import numpy as np
import onnxruntime as ort

GENESIS = bytes.fromhex(
    "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b2"
    "7ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
K = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
     0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
     0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
     0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
     0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
     0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
     0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
     0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]
IV = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
M = 0xFFFFFFFF


def compress(state, block):
    r = lambda x, n: ((x >> n) | (x << (32 - n))) & M
    w = list(struct.unpack(">16I", block))
    for i in range(16, 64):
        s0 = r(w[i-15], 7) ^ r(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = r(w[i-2], 17) ^ r(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & M)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + (r(e,6) ^ r(e,11) ^ r(e,25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) & M
        t2 = ((r(a,2) ^ r(a,13) ^ r(a,22)) + ((a & b) ^ (a & c) ^ (b & c))) & M
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M, c, b, a, (t1 + t2) & M
    return [(x + y) & M for x, y in zip(state, [a, b, c, d, e, f, g, h])]


MID = np.array(compress(IV, GENESIS[:64]), dtype=np.uint32)
BLK = np.array(struct.unpack(">3I", GENESIS[64:76]), dtype=np.uint32)
GNONCE = struct.unpack("<I", GENESIS[76:80])[0]


def expected(nonces):
    out = []
    for n in nonces:
        h = hashlib.sha256(hashlib.sha256(GENESIS[:76] + struct.pack("<I", int(n))).digest()).digest()
        out.append(struct.unpack(">8I", h))
    return np.array(out, dtype=np.uint32)


def to_bits(a): return ((a.astype(np.uint64)[..., None] >> np.arange(32, dtype=np.uint64)) & 1).astype(np.float32)
def from_bits(b): return (np.rint(b.reshape(b.shape[0], 8, 32)).astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1).astype(np.uint32)


LANE = np.uint64(1) << np.arange(32, dtype=np.uint64)


def pack_bs(words):
    """[N] uint32 values -> bitsliced [32, N/32]: row k, element m, bit j = bit k of value m*32+j"""
    v = words.astype(np.uint64).reshape(-1, 32)                                   # [M, 32 lanes]
    bits = (v[None, :, :] >> np.arange(32, dtype=np.uint64)[:, None, None]) & 1   # [32 k, M, 32 j]
    return (bits * LANE).sum(-1).astype(np.uint32)


def unpack_bs(t):
    """[8, 32, M] bitsliced -> [N, 8] uint32"""
    v = t.astype(np.uint64)
    lanes = (v[:, :, :, None] >> np.arange(32, dtype=np.uint64)) & 1              # [8, 32 k, M, 32 j]
    words = (lanes * (np.uint64(1) << np.arange(32, dtype=np.uint64))[None, :, None, None]).sum(1)  # [8, M, 32]
    return words.reshape(8, -1).T.astype(np.uint32)


def const_bs(words):
    return np.array([[[0xFFFFFFFF if (int(w) >> k) & 1 else 0] for k in range(32)] for w in words], dtype=np.uint32)


def feeds(kind, nonces):
    if kind == "u32": return {"mid": MID, "blk": BLK, "nonce": nonces.astype(np.uint32)}
    if kind == "bits": return {"mid_bs": const_bs(MID), "blk_bs": const_bs(BLK), "nonce_bs": pack_bs(nonces)}
    return {"mid_bits": to_bits(MID), "blk_bits": to_bits(BLK), "nonce_bits": to_bits(nonces.astype(np.uint32))}


def run(sess, kind, nonces):
    out = sess.run(None, feeds(kind, nonces))[0]
    if kind == "u32": return out
    return unpack_bs(out) if kind == "bits" else from_bits(out)


def session(kind, provider):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if provider == "DmlExecutionProvider":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    t = time.perf_counter()
    s = ort.InferenceSession(f"sha256d_{kind}.onnx", so, providers=[provider, "CPUExecutionProvider"])
    return s, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kinds", nargs="*", default=["u32", "relu"])
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--dml", action="store_true")
    ap.add_argument("--sizes", default="1024,16384"); ap.add_argument("--secs", type=float, default=8)
    a = ap.parse_args()
    provs = (["CPUExecutionProvider"] if a.cpu or not a.dml else []) + (["DmlExecutionProvider"] if a.dml else [])
    for kind in a.kinds:
        for prov in provs:
            tag = f"{kind:4s} on {prov.replace('ExecutionProvider', '')}"
            sess, t_load = session(kind, prov)
            # correctness: 256 nonces around the genesis nonce, every word of every hash vs hashlib
            nonces = (np.arange(256, dtype=np.int64) + GNONCE - 100).astype(np.uint32)
            got, want = run(sess, kind, nonces), expected(nonces)
            ok = np.array_equal(got, want)
            winners = nonces[got[:, 7] == 0]
            print(f"{tag}: load {t_load:.1f}s | 256 hashes match hashlib: {'OK' if ok else 'FAIL'}"
                  f" | genesis nonce found: {'OK' if list(winners) == [GNONCE] else winners}")
            if not ok:
                bad = np.nonzero((got != want).any(1))[0][:3]
                for i in bad: print("   nonce", hex(nonces[i]), "got", [hex(x) for x in got[i]], "want", [hex(x) for x in want[i]])
                continue
            for n in [int(x) for x in a.sizes.split(",")]:
                nonces = np.arange(n, dtype=np.uint32)
                run(sess, kind, nonces)  # warm-up
                t0, done = time.perf_counter(), 0
                while time.perf_counter() - t0 < a.secs:
                    run(sess, kind, nonces); done += n
                rate = done / (time.perf_counter() - t0)
                print(f"   batch {n:>8}: {rate/1e3:10.1f} kH/s")


if __name__ == "__main__":
    main()
