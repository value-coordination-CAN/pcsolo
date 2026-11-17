/* addbench - 32-bit modular addition micro-benchmark on an OpenCL GPU.
 *
 * Compares four ways to add uint32 numbers, all exact:
 *   native  : hardware integer add, 1 value per thread
 *   ripple  : bitsliced ripple-carry adder, 32 values per thread (one per bit lane), 31 dependent carry stages
 *   ks      : bitsliced Kogge-Stone parallel-prefix adder, carry depth log2(32) = 5, most gates
 *   bk      : bitsliced Brent-Kung parallel-prefix adder, depth 2*log2(32)-1 = 9, fewer gates than KS
 *
 * Workload: R dependent additions per value, Fibonacci-style (t = a + b; a = b; b = t), so nothing can be
 * folded away. Results are checked exactly against a CPU reference. Throughput is reported as additions per
 * second (lane-adds/s), swept over the number of threads to expose latency-bound vs throughput-bound regimes.
 *
 * Build (MSYS2 UCRT64): gcc -O2 -o addbench.exe addbench.c
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R 1023   /* additions per chain (multiple of 3 for the bitsliced buffer rotation) */

typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong; typedef uint64_t cl_bitfield;
typedef struct _cl_platform_id *cl_platform_id; typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context; typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem; typedef struct _cl_program *cl_program; typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;
#define CLAPI __stdcall
static cl_int (CLAPI *p_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
static cl_int (CLAPI *p_clGetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
static cl_int (CLAPI *p_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
static cl_context (CLAPI *p_clCreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
static cl_command_queue (CLAPI *p_clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int *);
static cl_program (CLAPI *p_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
static cl_int (CLAPI *p_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
static cl_int (CLAPI *p_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
static cl_kernel (CLAPI *p_clCreateKernel)(cl_program, const char *, cl_int *);
static cl_int (CLAPI *p_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_uint, size_t, void *, size_t *);
static cl_mem (CLAPI *p_clCreateBuffer)(cl_context, cl_bitfield, size_t, void *, cl_int *);
static cl_int (CLAPI *p_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
static cl_int (CLAPI *p_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
static cl_int (CLAPI *p_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
static cl_int (CLAPI *p_clFinish)(cl_command_queue);

static const char *SRC =
"#define R 1023\n"
"/* seeds for chain c */\n"
"#define SEEDA(c) ((c) * 2654435761u + 1u)\n"
"#define SEEDB(c) ((c) ^ 0xdeadbeefu)\n"
"__kernel void add_native(__global uint *out) {\n"
"  uint c = get_global_id(0), a = SEEDA(c), b = SEEDB(c);\n"
"  for (int i = 0; i < R; i++) { uint t = a + b; a = b; b = t; }\n"
"  out[c] = b;\n"
"}\n"
"/* transpose 32 chains (lanes) into bitsliced words: w[k] bit j = bit k of value for chain base+j */\n"
"#define LOAD_BS(A, B) \\\n"
"  uint base = get_global_id(0) * 32u; \\\n"
"  _Pragma(\"unroll\") for (int k = 0; k < 32; k++) { A[k] = 0; B[k] = 0; } \\\n"
"  for (int j = 0; j < 32; j++) { uint va = SEEDA(base + j), vb = SEEDB(base + j); \\\n"
"    _Pragma(\"unroll\") for (int k = 0; k < 32; k++) { A[k] |= ((va >> k) & 1u) << j; B[k] |= ((vb >> k) & 1u) << j; } }\n"
"#define STORE_BS(B) \\\n"
"  _Pragma(\"unroll\") for (int k = 0; k < 32; k++) out[get_global_id(0) * 32u + k] = B[k];\n"
"\n"
"/* ripple-carry: sum = x^y^c, carry = maj(x,y,c) -> one LOP3 each; 31 dependent stages */\n"
"inline void add_rc(uint *d, const uint *x, const uint *y) {\n"
"  uint c = x[0] & y[0]; d[0] = x[0] ^ y[0];\n"
"  #pragma unroll\n"
"  for (int k = 1; k < 32; k++) { uint s = x[k] ^ y[k] ^ c; if (k < 31) c = bitselect(x[k], y[k], x[k] ^ c); d[k] = s; }\n"
"}\n"
"/* Kogge-Stone: every position combines with position k-d at each of 5 levels */\n"
"inline void add_ks(uint *d, const uint *x, const uint *y) {\n"
"  uint G[32], P[32], P0[32];\n"
"  #pragma unroll\n"
"  for (int k = 0; k < 32; k++) { G[k] = x[k] & y[k]; P[k] = P0[k] = x[k] ^ y[k]; }\n"
"  #pragma unroll\n"
"  for (int dd = 1; dd < 32; dd <<= 1) {\n"
"    #pragma unroll\n"
"    for (int k = 31; k >= dd; k--) { G[k] |= P[k] & G[k - dd]; if (dd < 16) P[k] &= P[k - dd]; }\n"
"  }\n"
"  d[0] = P0[0];\n"
"  #pragma unroll\n"
"  for (int k = 1; k < 32; k++) d[k] = P0[k] ^ G[k - 1];\n"
"}\n"
"/* Brent-Kung: up-sweep builds prefixes at 2^i-1, down-sweep fills the rest; ~2x fewer prefix nodes than KS */\n"
"inline void add_bk(uint *d, const uint *x, const uint *y) {\n"
"  uint G[32], P[32], P0[32];\n"
"  #pragma unroll\n"
"  for (int k = 0; k < 32; k++) { G[k] = x[k] & y[k]; P[k] = P0[k] = x[k] ^ y[k]; }\n"
"  #pragma unroll\n"
"  for (int dd = 1; dd < 32; dd <<= 1) {\n"
"    #pragma unroll\n"
"    for (int k = 2 * dd - 1; k < 32; k += 2 * dd) { G[k] |= P[k] & G[k - dd]; P[k] &= P[k - dd]; }\n"
"  }\n"
"  #pragma unroll\n"
"  for (int dd = 8; dd >= 1; dd >>= 1) {\n"
"    #pragma unroll\n"
"    for (int k = 3 * dd - 1; k < 32; k += 2 * dd) G[k] |= P[k] & G[k - dd];\n"
"  }\n"
"  d[0] = P0[0];\n"
"  #pragma unroll\n"
"  for (int k = 1; k < 32; k++) d[k] = P0[k] ^ G[k - 1];\n"
"}\n"
"#define FIB_KERNEL(NAME, ADD) \\\n"
"__kernel void NAME(__global uint *out) { \\\n"
"  uint X0[32], X1[32], X2[32]; LOAD_BS(X0, X1) \\\n"
"  for (int i = 0; i < R; i += 3) { ADD(X2, X0, X1); ADD(X0, X1, X2); ADD(X1, X2, X0); } \\\n"
"  STORE_BS(X1) }\n"
"FIB_KERNEL(add_ripple, add_rc)\n"
"FIB_KERNEL(add_kogge, add_ks)\n"
"FIB_KERNEL(add_brent, add_bk)\n";

static uint32_t ref_chain(uint32_t c) {
    uint32_t a = c * 2654435761u + 1u, b = c ^ 0xdeadbeefu;
    for (int i = 0; i < R; i++) { uint32_t t = a + b; a = b; b = t; }
    return b;
}

static double now_ms(void) {
    static LARGE_INTEGER f; LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    HMODULE h = LoadLibraryA("OpenCL.dll");
    if (!h) { printf("OpenCL.dll not found\n"); return 1; }
#define L(f) if (!(*(FARPROC *)&p_##f = GetProcAddress(h, #f))) { printf("missing " #f "\n"); return 1; }
    L(clGetPlatformIDs) L(clGetDeviceIDs) L(clGetDeviceInfo) L(clCreateContext) L(clCreateCommandQueue)
    L(clCreateProgramWithSource) L(clBuildProgram) L(clGetProgramBuildInfo) L(clCreateKernel)
    L(clGetKernelWorkGroupInfo) L(clCreateBuffer) L(clSetKernelArg) L(clEnqueueNDRangeKernel)
    L(clEnqueueReadBuffer) L(clFinish)

    cl_platform_id plat; cl_uint np; cl_device_id dev; cl_uint nd; cl_int e;
    if (p_clGetPlatformIDs(1, &plat, &np) || p_clGetDeviceIDs(plat, 1 << 2, 1, &dev, &nd)) { printf("no GPU\n"); return 1; }
    char name[128] = ""; p_clGetDeviceInfo(dev, 0x102B, sizeof name, name, NULL);
    cl_context ctx = p_clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    cl_command_queue q = p_clCreateCommandQueue(ctx, dev, 0, &e);
    cl_program prog = p_clCreateProgramWithSource(ctx, 1, &SRC, NULL, &e);
    double tb = now_ms();
    if ((e = p_clBuildProgram(prog, 1, &dev, "", NULL, NULL))) {
        static char log[32768]; p_clGetProgramBuildInfo(prog, dev, 0x1183, sizeof log, log, NULL);
        printf("build failed %d:\n%s\n", e, log); return 1;
    }
    printf("device: %s | kernel build %.1f s | R = %d dependent adds per value\n\n", name, (now_ms() - tb) / 1000, R);

    const char *kn[4] = {"add_native", "add_ripple", "add_kogge", "add_brent"};
    const char *label[4] = {"native uint32", "ripple-carry", "Kogge-Stone", "Brent-Kung"};
    const int lanes[4] = {1, 32, 32, 32};
    cl_kernel k[4];
    size_t maxItems = 393216;
    cl_mem out = p_clCreateBuffer(ctx, 1, maxItems * 32 * 4, NULL, &e);
    uint32_t *host = malloc(maxItems * 32 * 4);

    printf("%-14s %10s  %s\n", "adder", "spill B/thr", "correctness (all lanes of 256 threads vs CPU)");
    for (int i = 0; i < 4; i++) {
        k[i] = p_clCreateKernel(prog, kn[i], &e);
        p_clSetKernelArg(k[i], 0, sizeof(cl_mem), &out);
        cl_ulong priv = 0; p_clGetKernelWorkGroupInfo(k[i], dev, 0x11B4, sizeof priv, &priv, NULL); /* PRIVATE_MEM_SIZE */
        size_t items = 256;
        p_clEnqueueNDRangeKernel(q, k[i], 1, NULL, &items, NULL, 0, NULL, NULL);
        p_clEnqueueReadBuffer(q, out, 1, 0, items * lanes[i] * 4, host, 0, NULL, NULL);
        int bad = 0;
        for (size_t t = 0; t < items; t++) {
            if (lanes[i] == 1) { if (host[t] != ref_chain((uint32_t)t)) bad++; continue; }
            for (int j = 0; j < 32; j++) {
                uint32_t v = 0;
                for (int b = 0; b < 32; b++) v |= ((host[t * 32 + b] >> j) & 1u) << b;
                if (v != ref_chain((uint32_t)(t * 32 + j))) bad++;
            }
        }
        printf("%-14s %10llu  %s (%zu values)\n", label[i], (unsigned long long)priv, bad ? "FAIL" : "exact", items * lanes[i]);
        if (bad) return 1;
    }

    /* throughput vs number of threads (work-items): low thread counts are latency-bound */
    const size_t sweep[] = {192, 1536, 12288, 98304, 393216, 1572864, 6291456};
    const int NS = 7;
    printf("\nadditions per second (G lane-adds/s) by number of GPU threads:\n%-14s", "adder");
    for (int s = 0; s < NS; s++) printf("%11zu", sweep[s]);
    printf("\n");
    for (int i = 0; i < 4; i++) {
        printf("%-14s", label[i]);
        for (int s = 0; s < NS; s++) {
            size_t items = sweep[s];
            /* bitsliced launches beyond 393K threads would exceed Windows' 2 s GPU watchdog */
            if (lanes[i] > 1 && items > maxItems) { printf("%11s", "-"); continue; }
            p_clEnqueueNDRangeKernel(q, k[i], 1, NULL, &items, NULL, 0, NULL, NULL); p_clFinish(q);   /* warm-up */
            int n = 0; double t0 = now_ms(), t;
            do { p_clEnqueueNDRangeKernel(q, k[i], 1, NULL, &items, NULL, 0, NULL, NULL); p_clFinish(q); n++; t = now_ms() - t0; }
            while (t < 700 && n < 2000);
            double adds = (double)items * lanes[i] * R * n;
            printf("%11.2f", adds / (t / 1000) / 1e9);
        }
        printf("\n");
    }
    return 0;
}
