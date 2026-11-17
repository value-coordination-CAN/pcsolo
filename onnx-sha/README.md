# SHA-256d as an ONNX graph: neural accelerator as a Boolean engine

An experiment: express Bitcoin's SHA-256d mining hash as an ONNX model, including a version built only from neural-network operations (MatMul, Add, Mul, ReLU). The aim is to see what it costs to run it on ML hardware such as a Hailo-8.

## The three graphs

`gen_sha256d.py` writes all three. Inputs are the midstate (8 words) and header words 16–18, which stay fixed for a mining job, plus a **batch of N nonces**. The output is the SHA-256d state for every nonce.

| Graph | Nodes | Ops |
|---|---|---|
| `sha256d_u32.onnx` | 6,479 | native uint32 `Add`, `BitShift`, `BitwiseAnd/Or/Xor/Not` |
| `sha256d_relu.onnx` | 37,101 | **only** `MatMul` 6k, `Relu` 8k, `Add` 17.6k, `Sub` 3.4k, `Mul` 2k (+ 11 Gather, 1 Concat) |
| `sha256d_bits.onnx` | 25,928 | **bit-packed Boolean network**: only `BitwiseAnd` 12.3k, `Gather` 6.9k, `BitwiseOr` 3.4k, `BitwiseXor` 3.4k |

How the bit-packed graph works:
- **Layout:** each SHA word is a uint32 tensor `[32, M]`. Row *k* is bit *k* of the word, and each element packs **32 different nonces** (one per bit lane). One node handles every bit position of 32·M nonces at once.
- **Rotations, shifts, byte-swap:** one `Gather` along the bit axis, plus an `And` mask for shifts.
- **Σ/σ, Ch, Maj:** plain Xor/And/Or.
- **Addition:** carry-save adders (Xor3 + Maj, carry shifted up one row), then Kogge-Stone along the bit axis (5 levels).
- **Constants:** constant words are `[32, 1]` rows of 0 / 0xFFFFFFFF that broadcast.

Memory traffic is 4 bytes per nonce per value, versus 128 bytes for the float ReLU graph.

How the ReLU graph computes Boolean logic on 0/1 values:
- **Words:** a 32-bit word is a 32-element bit vector.
- **Rotations, shifts, byte-swap:** 32×32 permutation matrices (MatMul).
- **Σ/σ functions:** XOR of three rotations, done as one MatMul with the summed matrix, then parity of s ∈ {0..3}: `s − 2·ReLU(s−1) + 2·ReLU(s−2)`.
- **Ch(e,f,g)** = `ReLU(e+f−1) + ReLU(g−e)`. **Maj** = carry of a 3-bit sum = `ReLU(s−1) − ReLU(s−2)`.
- **Addition:** carry-save adders (3→2), then a **Kogge-Stone** carry-lookahead adder (5 prefix levels instead of a 32-step ripple). Group-generate and group-propagate are mutually exclusive, so the OR in the prefix step is just a plain `Add`.
- **Constants:** padding and zero message words are folded at build time, so they never become graph nodes.

## Results (2026-11-16, Xeon E-2286M + NVIDIA T400, ONNX Runtime 1.24.4)

Correctness: on every provider, all 256 SHA-256d outputs around the genesis nonce are **bit-exact against `hashlib`**, and the genesis nonce is found.

| Implementation | Hardware | Throughput | vs native GPU |
|---|---|---|---|
| pcsolo OpenCL kernel | T400 | **216,000 kH/s** | 1× |
| pcsolo AVX2 | CPU, 8 threads | 55,000 kH/s | 1/4 |
| ONNX `u32` | CPU | 771 kH/s (batch 256k) | 1/280 |
| ONNX `u32` | T400 (DirectML) | 834 kH/s (batch 1M) | 1/260 |
| ONNX **ReLU** | CPU | 4.0 kH/s (batch 16k) | 1/54,000 |
| ONNX **ReLU** | T400 (DirectML) | 5.1 kH/s (batch 128k) | 1/42,000 |
| ONNX **bit-packed** | CPU | 189 kH/s (batch 256k) | 1/1,400 |
| ONNX **bit-packed** | T400 (DirectML) | 99 kH/s (batch 1M) | 1/2,600 |

**Bit-packed vs ReLU:** 47× faster on the CPU and 19× on the T400, from the same Boolean logic, just packed into integers instead of floats. It is still 4× (CPU) to 8× (DirectML) slower than the plain `u32` graph:
- each 32-bit add, a single `Add` node in the `u32` graph, expands into ~30 nodes;
- every node is a separate pass over memory.

(Bit-packed numbers: measured 2026-11-17 on an idle machine; 4M-nonce DirectML batches drop to 85 kH/s.)

All ONNX figures were measured on an idle machine. A first run while `pcsolo-gpu` was mining gave DirectML numbers about 2–2.5× lower and CPU numbers 5–20% lower, because the GPU and cores were shared.
The profile shows every node runs on DirectML (u32: 6,275 kernels; ReLU: 23,017 after ORT fusion), so there is no CPU fallback.

### Why the graph versions are so slow

The native kernel keeps the whole hash in **registers**. Each nonce costs about 2,500 instructions and never touches memory.

The graph versions run each node as a separate kernel that reads and writes a full `[N, 32]` tensor in memory:
- ~23k kernels per batch.
- At N = 32k, that's about 4 MB per tensor, so hundreds of GB of memory traffic per batch.
- The work is **memory-bound and dispatch-bound**, not compute-bound. The 32×32 MatMuls are far too small to keep matrix hardware busy.

The finding: as a neural network, SHA-256 is correct and runs anywhere, but it is roughly 40,000× slower than native integer code on the same chip.

## Hybrid test: native + ONNX engine on separate nonce ranges, same machine (2026-11-17)

Question: does adding the ONNX (bit-packed) engine *next to* native pcsolo add useful work?
Each engine hashes its own nonces, so the useful total is native + ONNX. Tool: `onnx_worker.py` (logs kH/s every 2 s). The ONNX rate is averaged over exactly the window where both ran.

| Setup | Native alone | Native + ONNX running | ONNX alone | ONNX while native | **Net change** |
|---|---|---|---|---|---|
| T400 only: pcsolo GPU vs ONNX on DirectML | 248.6 MH/s | 111.9 MH/s | 140.5 kH/s | 100.1 kH/s | **−136.6 MH/s (−55%)** |
| CPU only: pcsolo 14 threads vs ONNX on CPU | 65.9 MH/s | 27.2 MH/s | 155.4 kH/s | 140.1 kH/s | **−38.6 MH/s (−59%)** |
| Full rig: pcsolo 14 threads + GPU vs ONNX on DirectML | 314.3 MH/s | 175.0 MH/s | 144.8 kH/s | 102.7 kH/s | **−139.2 MH/s (−44%)** |

On shared silicon the hybrid **loses ~1,360 native hashes per ONNX hash gained on the GPU, and ~280 per hash on the CPU**.
- **GPU:** the Windows GPU scheduler gives each process a fair share of time, regardless of how efficiently it uses it.
- **CPU:** ONNX Runtime runs at normal priority on every core, while pcsolo deliberately runs at idle priority.

A parallel hybrid only adds work when the second engine runs on silicon that would otherwise be idle, e.g. a Hailo-8 on a separate Pi. Connect it to the pool as its own worker; the pool gives each connection its own extranonce, so the nonce ranges never overlap.

## Path to the Hailo-8 (Raspberry Pi AI HAT)

1. **Compile on x86 Linux, not the Pi.** Hailo's Dataflow Compiler (DFC) runs on Ubuntu x86-64, or WSL2 on this PC. It turns ONNX into a `.hef`. The Pi only runs it via HailoRT.
2. **Reshape for a CNN compiler.** Hailo expects image-like tensors. The natural mapping is "each pixel is a nonce, the channels are bits". Every 32×32 MatMul then becomes a **1×1 convolution** and the batch becomes the image (e.g. 256×256 = 65,536 nonces per frame).
3. **Fuse into wide layers.** 37k tiny nodes won't fit. The next generator should pack all live values (state, message schedule, carries) into one channel vector. Each step then becomes a single `Conv1x1 + ReLU` layer, giving a network about 1,500 layers deep instead of 37k nodes.
4. **Exact quantization.** Hailo runs INT8/INT4. Every value here is a small integer (0…5), so it can be exact, but only if the per-layer scales are integers. A wrong scale silently corrupts hashes. Safer versions use thresholds with ±0.5 margins (`ReLU(s−0.5)`-style) and "snap" layers.
5. **Expected result.** A dense 1×1 conv over ~800 channels is about 0.6M MACs per layer, so about 1G MACs per hash. At 26 TOPS that's about **10 kH/s at the theoretical best**, likely less, because the compiler will need many contexts for a network that deep. In other words, about 20,000× below the T400.

The Hailo is interesting for **hashes per joule**, at about 2.5 W, and as a stress test of how NPU compilers handle deep, integer-exact logic networks. It cannot contribute meaningfully to mining.

## Run

```
py -3.13 -m venv .venv
.venv\Scripts\pip install numpy onnx onnxruntime-directml
.venv\Scripts\python gen_sha256d.py                 # writes all three .onnx files
.venv\Scripts\python test_bench.py u32 relu bits --cpu   # verify + benchmark on CPU
.venv\Scripts\python test_bench.py u32 relu bits --dml   # ... on the GPU via DirectML
```

Python 3.15 has no onnxruntime wheels yet, so use 3.13.
