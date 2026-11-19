# pcsolo — Bitcoin solo "lottery" miner for Windows (CPU + GPU)

A small native miner that connects to a solo pool over Stratum v1 and hashes on your CPU and, optionally, your GPU.
- **CPU:** 8-way AVX2 SHA-256d, with a scalar fallback.
- **GPU:** OpenCL. It loads `OpenCL.dll` from your graphics driver at runtime, so there is nothing to install.

If it ever finds a block, almost the whole reward (3.125 BTC + fees, minus a small pool fee) goes to **your address**.

> **Reality check (this PC: Xeon E-2286M + NVIDIA T400), v1.4:**
> - GPU ≈ 280 MH/s with `--version-roll` (≈ 255 without)
> - CPU ≈ 60–75 MH/s
> - **`--gpu --version-roll` (thread count auto-tuned) ≈ 330–350 MH/s together**
>
> At network difficulty ~135T the expected time to a block is **~53 million years**. This is a lottery ticket, not income.

**Background:** why this miner exists, and what its numbers say about "one CPU, one vote": [Eighteen Years On, Satoshi's 2008 Promise Has a New Delivery Date: Roughly 640,000 Human Lifetimes Away](https://claude.ai/artifact/CfuUmCvUMjKm83C8xxfDWH).

## Run (PowerShell)

```powershell
& "C:\path\to\pcsolo.exe" --address <your BTC address> --gpu --version-roll
```

`&` is required in PowerShell when the path is in quotes. Only ever give it your **public receive address**.

| Option | Meaning |
|---|---|
| `-a, --address <addr>` | your BTC address (`bc1…`, `1…`, `3…`) |
| `-w, --worker <name>` | worker name on the pool (default `pc`) |
| `-p, --pool <host:port>` | or preset `public-pool` (default, `public-pool.io:21496`) / `ckpool` (`solo.ckpool.org:3333`) |
| `-t, --threads <n\|auto>` | CPU threads. **Default `auto`** (see *Auto-tuned thread count* below). A number fixes the count; `0` = GPU only. |
| `-g, --gpu` | also mine on the GPU |
| `--gpu-intensity <n>` | 2^n nonces per GPU launch, 16–28. Default 22 (~16 ms per launch). Launches are queued 4 deep, so 21–25 all run at full speed. Lower values keep the desktop smoother. |
| `--gpu-device <i>` | choose a GPU by index (default: first NVIDIA) |
| `--diff <d>` | share difficulty to suggest (public-pool uses 1 regardless) |
| `--version-roll` | GPU: BIP 310 version rolling. 8 header versions share one message schedule per nonce: **+13% GPU**. Needs pool support (public-pool has it); falls back to the normal kernel if the pool refuses. |
| `--versions <n>` | versions per nonce: 1, 2, 4 or **8** (default). Implies `--version-roll`. |
| `--scalar` | disable AVX2 |
| `--bitslice` | **experiment, slower:** bitsliced kernels (AVX2: 256 nonces per vector; GPU: 32 nonces per `uint`). See below. |
| `--normal-priority` | by default the process runs at **Idle** priority so the PC stays usable |
| `--bench [secs]` | offline hashrate benchmark (combine with `--gpu`, `-t`) |
| `--selftest` | checks every hashing path against reference SHA-256d (add `--gpu` to include the GPU) |

Every 15 s it prints the hashrate (CPU/GPU split), accepted/rejected shares, best difficulty and expected time to a block. Ctrl+C prints a summary.

## Did I win?

- The console prints `!!!!!!!! BLOCK FOUND !!!!!!!!`, and the block is submitted automatically.
- The pool dashboard: `https://web.public-pool.io/#/app/<your address>`
- The definitive check is your address on `https://mempool.space/address/<your address>`. A win is an incoming coinbase payment of ~3.1+ BTC, spendable after 100 confirmations (~17 h).
- Shares and "best diff" only show the miner is working. A block needs a best diff of about 135T.

## Experiment: carry the nonce across templates (`--carry-nonce`)

By default each worker starts its nonce counter at 0 on every new block template. `--carry-nonce` instead continues the counter from where it was when the template changed.

**It does not change your odds, and that is the point of including it.** A new template is fresh data (new previous-block hash, new merkle root), so nonce *N* in the old template and nonce *N* in the new one give unrelated hashes. Every nonce is equally likely regardless of where the counter starts, and the miner already never repeats work: the extranonce counter only ever increases, so each search uses unique block data whether the nonce resets to 0 or not. Starting high just means the low nonces of that template go unchecked instead — the same number of guesses, equally good.

Added as a measurable control: run it for a day against the default and the share-rate and best-diff distributions are statistically identical. The only lever on time-to-block is hashes per second.

## Watching new blocks

Each time anyone on the network finds a block, the pool sends work for the next one, and pcsolo logs it:

```
[21:22:50] New block template for block 976489 | network difficulty 134.82T | previous block took 8m 12s
```

- The block number is the one your miner is now trying to find. It is read from the job itself (BIP 34 puts the height in the payout transaction).
- "previous block took" is the time between two templates as seen by this miner, so it appears from the second block on. The average is 10 minutes, with large swings.

## Is the pool paying me? (payout check)

You don't have to trust the pool about who gets paid. Every job contains the payout ("coinbase") transaction, and the miner hashes it into every header it tries. So the payout is fixed before a block is found, and the pool can't change it afterwards.

On every new block template, pcsolo decodes that transaction and compares its outputs with your `--address`:

```
[18:04:48] New block template | network difficulty 134.82T
[18:04:48] Payout check: 3.14986205 BTC (100.00%) to YOUR address
```

- If part of the reward goes elsewhere (a pool fee), the line shows how much and to how many other outputs.
- If nothing goes to your address, it prints `Payout check: WARNING - this job pays NOTHING to your address`. Stop mining on that pool.
- It understands `bc1…` (segwit and taproot), `1…` and `3…` addresses.

Tested live on public-pool.io on 2026-11-18 with a `bc1q…` and a `1…` address: each run reported 100% to its own address.

What it can't detect: a pool that receives your winning block and doesn't broadcast it. The pool would gain nothing from that, because the payout inside is yours.

## How it's optimised

- **Midstate:** the first 64 bytes of the header are hashed once per work unit.
- **Per-job precompute (v1.3):** in header block 2 only the nonce word changes. So the following are computed once per work unit on the CPU, instead of once per hash:
  - rounds 0–2 and the nonce-independent half of round 3
  - message-schedule words W16 and W17, and the constant parts of W18 and W19

  All three backends (scalar, AVX2, OpenCL) use it.
- **GPU launch queue (v1.3):** up to 4 kernel launches are kept in flight. The host only waits for the oldest while the newer ones keep the GPU busy, so the GPU no longer idles while results are read, new work is built, or the idle-priority process is briefly descheduled. `timeBeginPeriod(1)` gives 1 ms sleep accuracy, and the feeder sleeps through most of each launch instead of spinning on NVIDIA's blocking read.
- **Version rolling (v1.4, `--version-roll`), a CPU/GPU split per work unit.**
  - **How it works:** the pool lets the miner change version bits (BIP 310, mask `1fffe000`). A different version changes only header *block 1*. Block 2 (merkle tail, time, bits, nonce), and therefore the whole first-hash message schedule W18–W63, stays the same for all versions.
  - **CPU:** builds 4 versions per work unit (4 midstates and 4 precomputes), once per work unit.
  - **GPU:** expands the schedule once per nonce and runs it through the 4 midstates. Each version is a separate header, so each counts as a hash.
  - **Shares:** submitted with the version bits as the 6th `mining.submit` parameter.
  - **Launch size:** each launch covers a quarter of the nonces, so it takes as long as a normal launch.
  - This is the idea behind ASICs' "overt AsicBoost".
- **GPU kernel:** fully unrolled. `rotate()`/`bitselect()` compile to Turing's funnel-shift and LOP3 instructions, and the unused final rounds are dropped.

## Build

The exe is static (no DLLs) and runs on any x64 Windows 10/11. GPU support needs a driver that provides OpenCL, which NVIDIA, AMD and Intel drivers all do. To rebuild with MSYS2 (UCRT64 gcc):

```
build.cmd
```

or `gcc -O3 -static -o pcsolo.exe pcsolo.c -lws2_32 -lwinmm`.

## Notes

- Mining with the GPU keeps it at 100% load (the T400 settles around 74–82 °C, fan up). Use `-t 0` or drop `--gpu` if that bothers you.
- Only run it on hardware and electricity you own. Many employers forbid mining on company machines.
- Some antivirus tools flag any crypto miner. That is expected for this kind of program.

## Benchmarks (2026-11-17, idle machine, 10 s offline bench)

GPU only, by intensity: v1.2 (one launch at a time) vs v1.3 (launch queue + precompute):

| intensity | 21 | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|
| v1.2 | 67 | 135 | 180 | 216 | 245 MH/s |
| v1.3 | **255–260** | **255–258** | **256** | **255** | **252** MH/s |

CPU threads (v1.3, CPU only): 8 → 58, 12 → **65**, 16 → 61 MH/s. Hyperthreading helps a little, and the 16-thread result drops because the CPU throttles harder.
Precompute made no measurable CPU difference: the CPU is power- and heat-limited, so saving ~3% of the arithmetic doesn't show up.

CPU + GPU (v1.3): `-t 8` 310, `-t 14` **316–322**, `-t 15` 316, `-t 16` 305 MH/s. At 16 threads the GPU drops to 246 because its feeder thread gets starved.

## Auto-tuned thread count (v1.4, default `-t auto`)

At startup, before connecting to the pool, pcsolo runs a ~25 s offline calibration on the real hashing paths, with the GPU and version rolling active if enabled:
1. It measures **total** CPU+GPU throughput at a coarse set of thread counts (L/2, 3L/4, L−2, L−1, L, where L = logical cores).
2. It then measures the two neighbours of the best one.
3. It keeps the **fewest threads within 1% of the best rate**, so you get the same hash rate with less heat and power. Unused threads stay parked.

Because it measures total throughput, it automatically accounts for CPU heat limits and for the GPU feeder needing CPU time. Two runs on this PC (Xeon E-2286M, 16 logical cores, T400, version rolling):

| threads | 8 | 11 | 12 | 13 | 14 | 15 | 16 | picked |
|---|---|---|---|---|---|---|---|---|
| run 1 (MH/s) | 339.7 | 343.7 | **350.5** | 344.1 | 348.3 | 347.9 | 349.0 | 12 |
| run 2 (MH/s) | 338.2 | – | 344.1 | – | 346.0 | 344.7 | **346.7** | 12 |

From 12 to 16 threads the total is flat (within ~1%), so extra threads only add heat. Live with the auto-picked 12 threads on public-pool.io, 2 minutes: **320–334 MH/s, 9 shares accepted, 0 rejected**.

## Version count sweep, register profile and share log (v1.4)

**Sweep.** GPU only, 3 interleaved rounds × 10 s on a quiet GPU (2026-11-17). Registers, spills and occupancy come from NVIDIA's `-cl-nv-verbose` build log (`--selftest` / `--bench` force a fresh build so it's always reported).

| versions per nonce | median MH/s (min–max) | vs normal | registers | spill | warps/SM |
|---|---|---|---|---|---|
| normal kernel | 254.2 (250.0–258.8) | – | 42 | 0 | 32/32 |
| 1 | 252.1 (248.7–256.7) | −0.8% | 43 | 0 | 32/32 |
| 2 | 271.4 (269.3–275.6) | +6.8% | 80 | 0 | 25/32 |
| 4 | 281.9 (280.2–288.2) | +10.9% | 80 | 0 | 25/32 |
| **8 (default)** | **287.3 (284.8–291.1)** | **+13.0%** | 80 | 0 | 25/32 |

What it shows:
- **Diminishing returns, as predicted.** With the shared message schedule being a fraction *s* of a hash's work, the speed-up is 1 / (1 − s·(1 − 1/N)). The ×2 result gives s ≈ 12.7%. That predicts ×4 = +10.5% (measured +10.9%) and ×8 = +12.5% (measured +13.0%), and caps the gain at **+14.6%** however many versions are used.
- **No register cliff at 8.** Keeping the 46-word schedule live costs ~37 extra registers (43 → 80) as soon as there are 2+ versions. It doesn't grow with more versions, because versions run one after another. Nothing spills, and occupancy drops from 32 to 25 warps per SM without hurting throughput.
- ×1 runs through the version-rolling code path but shares nothing, so it confirms the constant-buffer path costs ≤1%.
- The earlier "+14% for ×4" A/B came out higher because its normal-kernel runs were slower (245.6). This sweep is the cleaner number.

**Measurement pitfall:** the first sweep attempt swung between 145 and 290 MH/s whatever the kernel, because Chrome was using ~17% of the GPU's 3D engine. Windows time-slices the GPU, so any rendering app steals mining time. Check GPU load before benchmarking.

**Share log.** Every submitted share is logged when the pool answers, with source (CPU/GPU), version index and version bits, nonce, and its exactly re-verified difficulty. The double SHA-256 is recomputed on the host from the exact rolled header. A GPU candidate that doesn't reproduce is counted as `failed re-verification` and never submitted. Ctrl+C prints accepted/rejected counts per version. Live example, ×8:

```
share #11 GPU v3 [bits 00006000] nonce 021aa699 diff 5.25 (hash re-verified) -> accepted
share #12 CPU nonce 06c40c12 diff 2.415 (hash re-verified) -> accepted
share #14 GPU v2 [bits 00004000] nonce 373a0502 diff 1.802 (hash re-verified) -> accepted
share #15 GPU v7 [bits 0000e000] nonce 3969a892 diff 1.708 (hash re-verified) -> accepted
```

In that 2-minute run, GPU shares from v1, v2, v3, v4, v6 and v7 were all accepted, with 0 rejected and 0 failed re-verifications. That confirms the pool reconstructs rolled versions correctly.

## Final GPU tuning round (2026-11-17)

Four remaining ideas were each built behind a flag, self-tested (exact (nonce, version) candidate match against the scalar path), then benchmarked GPU-only. They were interleaved with the ×8 baseline on a quiet GPU, 2–3 rounds of 10 s each.

| Variant | Registers / spill / warps | Avg MH/s | vs ×8 baseline | Kept? |
|---|---|---|---|---|
| ×8 baseline | 80 / 0 B / 25 | 289.3 | – | – |
| **×8 + second-hash precompute** | 80 / 0 B / 25 | **290.1** | **+0.3%** (higher in all 3 paired rounds) | **yes, default** (`--no-second-pre` disables) |
| ×16 versions | 89 / 0 B / 21 | 280.7 | −3.0% | no (`--versions 16` available) |
| ×16 + second-hash precompute | 90 / 0 B / 21 | 269.5 | −6.8% | no |
| register cap 72 | 72 / 24 B / 28 | 284.5 | −1.7% | no |
| register cap 64 | 64 / 56 B / 32 | 264.3 | −8.6% | no |
| workgroup 256 | 80 / 0 B / 25 | 283.8 | −1.9% | no |
| workgroup 128 | 80 / 0 B / 25 | 232.8 | −19.5% | no |
| workgroup 64 | 80 / 0 B / 25 | 234.5 | −19% | no |

Findings:
- **16 versions loses.** The model predicted only +0.5% over ×8. In practice registers rise to 89 and occupancy drops to 21 warps, and the 16-way unrolled kernel is ~2× larger, so it ends up 3% *slower*. ×8 is the sweet spot on the T400.
- **Second-hash precompute: almost nothing left to gain.** The second hash's first round starts from constants, and it can stop after round 60. Same registers, +0.3%. The GPU compiler was already folding most of it.
- **Register caps make things worse.** More occupancy doesn't pay when it costs spills: 64 registers gave full occupancy but spilled 56 B per thread and lost 8.6%. Occupancy isn't the bottleneck; arithmetic throughput is.
- **The driver's default workgroup size is best.** Fixed sizes of 64 or 128 cost ~19%.
- **SHA-NI isn't possible here.** A CPUID check shows the Xeon E-2286M (Coffee Lake) has no SHA-NI and no AVX-512, only AVX2.
- **Overlapping GPU execution and CPU preparation was already done** (the 4-deep launch queue).

Live with the final defaults (auto threads → 12, ×8 versions, second-hash precompute), 2 minutes on public-pool.io: **342–347 MH/s**, 7 shares accepted from versions v0/v3/v5/v6/v7, 0 rejected.

## Minimum-work audit: rounds 3–6 constant folding (v1.5, 2026-11-18)

Question: does the GPU kernel do only the SHA-256d work that depends on the nonce? Audit of the ×8 version-rolling kernel:

| Saving | Status |
|---|---|
| Midstate (header block 1) hashed once per job and version | already done |
| Rounds 0–2 and the constant half of round 3 precomputed on the host | already done |
| W16/W17 per-job constants, W18/W19 precomputed apart from their nonce term, zero padding words folded | already done |
| One message schedule shared by all 8 versions | already done |
| Second hash: fixed padding words folded, round 0 precomputed, stop after round 60 (only H7 is tested) | already done (v1.4) |
| Final test: one byte-swap + one compare; full 256-bit check on the CPU for rare candidates only | already done |
| **New:** in rounds 3–6 the `h` and `d` inputs are still per-job constants, so `h+K+W` and `d+h+K+W` come from the host | **v1.5, default** (`--no-extra-pre` disables) |

Not worth doing on this GPU: precomputing `K[16]+W16` and `K[17]+W17`. NVIDIA's 3-input add (IADD3) already absorbs the extra term, so it saves no instructions.

Result: self-test passes both ways (genesis found, 65/65 (nonce, version) candidates match the scalar path). Same 80 registers, 0 B spill, 25 warps. GPU-only, 8 interleaved rounds of 20 s:

| Rounds | `--no-extra-pre` | `--extra-pre` (new default) | Difference |
|---|---|---|---|
| 1–4 (GPU warming up 63 → 78 °C, new always ran second) | 286.8 MH/s | 285.2 MH/s | −0.6%, the same size as the warm-up drift between runs |
| 5–8 (warm, order reversed: new ran first) | 280.6 MH/s | **281.2 MH/s** | **+0.2%** (higher or equal in all 4 pairs) |

That is the predicted size: about 4 instructions saved out of roughly 1,500 per version-hash. It is at the edge of what the benchmark can resolve, because the rates come in steps of ~0.4 MH/s. The kernel is now at the practical minimum; what remains is the unavoidable ~61 + 60 SHA rounds and two message schedules per hash, and the 31 W power cap sets the speed.

## Occupancy and instruction-level parallelism (v1.5, 2026-11-18)

Two questions about the ×8 kernel:
- **Occupancy:** does 80 registers per thread (25 of 32 warps per SM) hold it back?
- **ILP:** would running independent hashes interleaved inside one thread hide the SHA-256 dependency chain better?

Experiment `--gpu-ilp 2`: the same kernel, but two header versions run in `uint2` lanes. Every SHA step then issues two independent instructions. It is self-tested (genesis, 65/65 (nonce, version) candidates match the scalar path) and benchmarked GPU-only after a 40 s warm-up.

| Variant | Registers / spill / warps per SM | MH/s |
|---|---|---|
| **×8, one version per pass (default)** | 80 / 0 B / 25 | **278.5** (median of 8) |
| ×8, two versions interleaved (`--gpu-ilp 2`) | 122 / 0 B / 16 | 272.9 (median of 8), **−1.6%**, lower in 7 of 8 alternating pairs |
| ILP×2 + register cap 96 | 96 / 36 B / 21 | ~272 (2 runs) |
| ILP×2 + register cap 80 | 80 / 100 B / 25 | ~270 (2 runs) |

What this shows, together with the earlier register-cap results (72 regs → 28 warps: −1.7%; 64 regs → 32 warps: −8.6%):
- **Occupancy is not the limit.** Going from 25 down to 16 warps per SM costs only ~1.6%, even with extra ILP added. Going up to 28–32 warps costs more because of spills. A Turing SM has 4 schedulers and ~4-cycle ALU latency, so 25 warps (about 6 per scheduler) already hide the dependency chain.
- **The kernel already has ILP.** Within each SHA round, the `t1` path (Σ1, Ch, +K+W) and the `t2` path (Σ0, Maj) are independent, and the message-schedule words can be computed ahead. The compiler interleaves these already.
- **The real limit is arithmetic throughput under the 31 W power cap** (throttle reason 0x4 is active during every run). More parallelism can't help once every ALU slot is in use; it only adds register pressure.

Default stays `--gpu-ilp 1`.

## Compiled instruction count and throughput ceiling (2026-11-18)

Instead of more kernel changes: how many machine instructions does a hash really cost, and how close is the T400 to its limit?

Method: `PCSOLO_KBIN=<file>` makes pcsolo dump the driver's compiled program, which is PTX text for `sm_75`. CUDA Toolkit 13.5 turns it into machine code and disassembles it: `ptxas -arch=sm_75 -o k.cubin k.ptx`, then `nvdisasm -c k.cubin`. `ptxas` reports the same register counts as the driver (80 and 42), so the code should match what the driver runs. Both kernels are straight-line code: the only branches skip the rare candidate-report block, so the static count is the executed count.

| Kernel | Instructions per work-item | Hashes per work-item | **Instructions per hash** |
|---|---|---|---|
| Normal (`search`) | 2,496 | 1 | **~2,475** |
| Version rolling ×8 (`search_vrn`) | 17,320 | 8 | **~2,150** |

- The ×8 kernel needs 13% fewer instructions per hash, which matches its measured +13% speed.
- Instruction mix of the ×8 kernel: 45% shifts/rotates (`SHF`), 26% 3-input logic (`LOP3`), 19% 3-input adds (`IADD3`), 8% 2-input adds placed on the multiplier unit (`IMAD.IADD`). Under 2% is anything else: there are no register moves and no wasted work to remove.

Ceiling, with the simple model "each of the T400's 384 cores executes one instruction per clock cycle":

| | Clock | Ceiling | Measured | Share of ceiling |
|---|---|---|---|---|
| Sustained while mining (power cap active, flag 0x4) | ~1,815 MHz | ~323 MH/s | 278–281 MH/s | **~87%** |
| GPU's maximum clock (`clocks.max.sm`) | 2,100 MHz | ~374 MH/s | – | ~75% |

What this means:
- **The kernel runs at about 87% of what the chip can do at its sustained clock.** The earlier rough guess (1,500 instructions, ceiling 460 MH/s, 60% used) was wrong: the real count is ~2,150.
- **The largest remaining gap is the clock, not the code.** The power cap holds the GPU ~14% below its maximum clock. That is a hardware and firmware limit of this laptop.
- **Software headroom is at most ~13%**, and part of that is unavoidable (launch gaps, warp scheduling, 25 of 32 warps resident).
- Efficiency: ~280 MH/s at the ~31 W cap is roughly **9 MH per joule**. The 31 W is the cap, not a reading, because `nvidia-smi` reports no power draw on this laptop. A current mining ASIC is several thousand times more efficient.

Caveats: the one-instruction-per-core-per-cycle model is a simplification, and the toolkit's `ptxas` may differ slightly from the one inside the driver.

## Version rolling benchmark (v1.4, 2026-11-17, idle machine)

Interleaved A/B, GPU only, 12 s each:

| run | 1 | 2 | 3 | avg |
|---|---|---|---|---|
| normal kernel | 248.1 | 246.0 | 242.8 | 245.6 MH/s |
| `--version-roll` | 281.4 | 280.0 | 278.9 | **280.1 MH/s (+14.0%)** |

The gain matches the estimate. Schedule expansion is ~450 of ~2,400 GPU instructions per hash, and 3 of every 4 expansions are saved. The second hash can't be shared, because its input differs per version.

Live on public-pool.io, `-t 14 --gpu --version-roll`, 2 minutes:
- the pool confirmed `version-rolling` with mask `1fffe000`
- **330–352 MH/s** (CPU 60–77 + GPU 270–278)
- **5 shares accepted, 0 rejected**

## Experiment: bitsliced kernels (`--bitslice`)

Bitslicing transposes the data. A SHA word becomes 32 machine words, and word *k* holds bit *k* for many nonces at once (256 per AVX2 vector, 32 per GPU `uint`). In that layout:
- Rotations and shifts become free re-indexing.
- XOR, Ch and Maj become plain bitwise operations.
- Every 32-bit addition becomes an explicit ripple-carry adder: 5 ops per bit, or 2 when one operand is constant.

Other details:
- Lanes are consecutive nonces, so the nonce's low byte lands in W3 bits 24–31 as fixed lane patterns, and no runtime transpose is needed.
- The state is ring-buffered (no copying), constant schedule words are folded, and the second hash stops after round 60.

Results are bit-exact: both kernels find the genesis nonce and return exactly the same candidate set as the scalar path in `--selftest`. Speed on an idle machine, 2026-11-17:

| | native | bitsliced | ratio |
|---|---|---|---|
| CPU, 1 thread | 12.0 MH/s | 4.9 MH/s | 0.41× |
| CPU, 8 threads | 54.5 MH/s | 20.7 MH/s | 0.38× |
| CPU, 12 threads | 59.2 MH/s | 22.8 MH/s | 0.38× |
| GPU T400, intensity 22 | 258 MH/s | 5.6 MH/s | 0.022× |
| GPU T400, intensity 24 | 255 MH/s | 1.4 MH/s | 0.005× |

**Why it loses.** SHA-256 is built from 32-bit additions, rotations and XORs.
- Bitslicing makes rotations free, but they were already cheap.
- It turns each 32-bit add, which the hardware does in one instruction with built-in carry logic, into ~32 dependent full adders.
- **CPU:** the bitsliced state (~27 KB) lives in L1 cache, not registers, so every gate also costs loads and stores. AVX2 has no 3-input logic instruction (that needs AVX-512's `vpternlogd`).
- **GPU:** each work-item needs ~3.4 KB of state (A/E rings, 16-word schedule, temporaries), far beyond the 255-register limit. Everything spills to slow local memory, and that dominates. Larger launches make it worse.

That matches the cryptography literature: bitslicing wins for designs built on lookup tables (e.g. DES), not for add-rotate-XOR designs like SHA-256. The native kernels remain the default.

## Verified (2026-11-17)

- `--gpu --selftest`:
  - SHA-256 "abc" vector and genesis-block SHA-256d.
  - The **precomputed path matches the reference SHA-256d word for word** on 64 nonces.
  - The scalar, AVX2 and GPU scans all find the genesis nonce, the GPU across a single 4M-nonce launch.
  - AVX2 and GPU return **exactly the same 246 candidates as the scalar path** on a 64K-nonce cross-check.
  - Target/difficulty maths correct.
- Live on public-pool.io, v1.3 with `-t 14 --gpu` for 2 minutes: **304–317 MH/s** (CPU ~60 + GPU ~248), **9 shares accepted, 0 rejected**.
