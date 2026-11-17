# 32-bit modular addition: ripple vs parallel-prefix vs native

This isolates the one operation that makes SHA-256 expensive as a Boolean network: `a + b mod 2^32`.

**Workload:** chains of dependent additions (`t = a + b; a = b; b = t`), so nothing can be folded away. Every result is checked **exactly** against a CPU reference.

| Adder | Logic | Carry depth | Gates per 32-bit add (approx.) |
|---|---|---|---|
| native `uint32` | hardware adder | (hardware) | 1 instruction |
| ripple-carry | sum = x⊕y⊕c, carry = maj(x,y,c) | 31 dependent stages | ~63 (one LOP3 each) |
| Kogge-Stone | G/P prefix, every position at every level | 5 levels | ~340 |
| Brent-Kung | up-sweep + down-sweep prefix tree | 9 levels | ~180 |

The Boolean adders are **bitsliced**: one 32-bit machine word holds the same bit position of 32 different values. So one gate instruction does 32 additions' worth of work for that bit.

## Regime 1 — fused kernels in registers (OpenCL, NVIDIA T400)

Source: `addbench.c`. Each value goes through 1,023 dependent adds. Figures are G additions/s by number of GPU threads, measured on an idle machine, 2026-11-17.

| Adder | Spill per thread | 192 | 1,536 | 12,288 | 98,304 | 393,216 | 1.5M | 6.3M |
|---|---|---|---|---|---|---|---|---|
| native uint32 | 0 B | 7.5 | 62 | 347 | 816 | **908** | 900 | 900 |
| ripple-carry | 0 B | **27.5** | **192** | 198 | 207 | 208 | – | – |
| Brent-Kung | 0 B | 13.6 | 99 | 102 | 104 | 104 | – | – |
| Kogge-Stone | **256 B** | 1.3 | 3.8 | 3.8 | 3.9 | 3.8 | – | – |

Bitsliced launches beyond 393K threads would exceed Windows' 2-second GPU watchdog. All four adders are exact on 8,192 values (256 threads × 32 lanes).

What it shows:
- **Register pressure decides first.** Kogge-Stone keeps G, P and P0 (96 words) live next to the 96 words of operands. That exceeds the register file, so it spills 256 B per thread to local memory and runs **54× slower than ripple**. Its log-depth carry is irrelevant once it spills.
- **Carry depth doesn't help, even when latency-bound.** At 192 threads (about one warp per GPU core) the GPU can't hide latency, so this is exactly where shallow depth should pay off. Ripple still beats Brent-Kung 2×. A single warp issues at most one instruction per cycle, so Brent-Kung's ~3× more instructions cost more than ripple's 31-stage carry chain.
- **Bitslicing beats native only at very low occupancy:** 3.7× at 192 threads and 3.1× at 1,536, because each thread does 32 additions. Once ~12K threads keep the GPU busy, native wins, peaking about **4.4× above ripple**.
- **Caveat on native:** 908 G adds/s is above the T400's ~700 G simple integer ops/s. The compiler probably merges consecutive Fibonacci steps (e.g. into 3-input adds), so treat native as an upper bound.

## Regime 2 — unfused graph (ONNX Runtime: CPU and DirectML on the T400)

Source: `../onnx-sha/add_bench.py`. Each node is a separate kernel and memory pass, the way unfused NPU graphs run. 16 dependent adds; figures are G additions/s.
- **ripple:** bit-packed `[32, M]` split into 32 rows of `[1, M]`, carry propagated row by row.
- **Kogge-Stone / Brent-Kung:** whole `[32, M]` tensors, shifts via `Gather` along the bit axis. Brent-Kung selects rows per level with masks.

| Adder | Nodes per add | CPU 4K | CPU 64K | CPU 1M | DML 4K | DML 64K | DML 256K | DML 1M | DML 8M |
|---|---|---|---|---|---|---|---|---|---|
| native `Add` | 1 | 1.29 | 2.68 | 4.38 | 0.150 | 1.42 | 2.12 | 2.64 | 1.10 |
| ripple | 154 | 0.034 | **0.316** | **0.587** | 0.002 | 0.030 | **0.107** | **0.337** | **0.756** |
| Kogge-Stone | 37 | **0.063** | 0.085 | 0.109 | **0.007** | **0.054** | 0.076 | 0.089 | 0.044 |
| Brent-Kung (masked) | 70 | 0.030 | 0.045 | 0.075 | 0.003 | 0.027 | 0.037 | 0.041 | 0.024 |
| ReLU Kogge-Stone (float bits) | 52 | – | – | 0.001* | – | – | – | 0.001* | – |

\* measured at 262,144 values; float bits use 32× the memory. All variants are exact on every run.

What it shows: **there's a crossover, and it's set by batch size.**
- **Small batches (overhead-bound):** each node costs a fixed dispatch overhead, so *node count* rules. Kogge-Stone's 37 nodes beat ripple's 154: **3.5× on DirectML at 4K, 1.8× at 64K**, and 1.9× on CPU at 4K. This is the only place parallel-prefix wins anything.
- **Large batches (memory-bound):** ripple's nodes each touch one bit row (1/32 of the tensor), about 5 tensor-equivalents of traffic per add. Kogge-Stone's 37 whole-tensor nodes move ~7× more data. Ripple wins **3.8× at 1M and 17× at 8M on DirectML**. At 8M, ripple is only 1.45× slower than native `Add`.
- **Brent-Kung never wins in tensor form.** Its gate savings disappear when each level still processes the whole tensor through masks. It has more levels than Kogge-Stone and no less work per level.
- **Float/ReLU form (what an INT8 NPU would run):** about 100× slower than bit-packed Kogge-Stone, because each bit is a float rather than a bit in a packed word.

## What this means for the T400 and the Hailo-8

- **T400:** native integer add wins whenever the GPU is reasonably busy. Among the Boolean adders, ripple wins: fewest instructions and lowest register pressure. Parallel-prefix carry depth isn't the bottleneck in either regime once lane parallelism exists.
- **Hailo-8 (not measured; reasoning from these results):** Hailo compiles a network into an on-chip dataflow pipeline, where *each layer consumes chip resources*. That is closer to the overhead-bound regime than the memory-bound one:
  - fewer, wider layers should matter more than total gate count;
  - ripple (32 sequential layers per add) would be the worst fit;
  - Kogge-Stone would be the best of the three.

  Hailo also has no bitwise ops, so it would have to use the float/ReLU form, the slowest row above. A real Hailo measurement needs the Dataflow Compiler (x86 Linux) and the device.

## Run

```
cd addbench
gcc -O2 -o addbench.exe addbench.c        (MSYS2 UCRT64)
addbench.exe

cd ../onnx-sha
.venv\Scripts\python add_bench.py --n 1048576          (CPU)
.venv\Scripts\python add_bench.py --dml --n 1048576    (DirectML)
```
