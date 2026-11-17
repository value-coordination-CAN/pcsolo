/* pcsolo - tiny Bitcoin solo "lottery" miner for Windows (CPU, Stratum v1), native C.
 * Build (MSYS2 UCRT64):  gcc -O3 -static -o pcsolo.exe pcsolo.c -lws2_32 -lwinmm
 * Uses an AVX2 8-way SHA-256d path when the CPU supports it, scalar otherwise.
 * Your odds of finding a block are astronomically small. This is for fun and learning. */
#define _WIN32_WINNT 0x0601
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <mmsystem.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <ctype.h>

#define VERSION "pcsolo/1.4"
#define MAXCB 4096
#define MAXBR 32
#define CHUNK 65536u

/* ======================= config / state ======================= */
static char host[256] = "public-pool.io";
static char portstr[16] = "21496";
static char address[128], worker[40] = "pc", user[180];
static int nthreads;                /* CPU miner threads created */
static volatile int activeThreads;  /* threads currently hashing; the rest are parked (auto-tune) */
static int autoThreads = 1;         /* -t auto (default) */
static volatile int offline;        /* bench / calibration: synthetic job, ignore candidates */
static double suggestDiff = 0.05;
static int useAvx2;

typedef struct {
    char id[128];
    uint8_t prev[32];
    uint8_t cb1[MAXCB], cb2[MAXCB]; int cb1len, cb2len;
    uint8_t br[MAXBR][32]; int nbr;
    uint8_t ver4[4], nbits4[4], ntime4[4];
    char ntimehex[16];
    uint8_t target[32];
    double netdiff;
    uint8_t en1[32]; int en1len, en2size;
} Job;

static SRWLOCK jobLock = SRWLOCK_INIT;
static Job gjob;
static volatile LONG jobGen, haveJob;
static uint8_t en1[32]; static int en1len, en2size = 4, en1ok;
static volatile double poolDiff = 1;
static volatile LONG64 totalHashes, en2Counter, submitId = 10;
static volatile LONG accepted, rejected, blocksFound;
static double bestDiff;
static CRITICAL_SECTION bestCs, sendCs, logCs;
static volatile SOCKET sock = INVALID_SOCKET;
static ULONGLONG startTick;

static const double DIFF1 = 65535.0 * 411376139330301510538742295639337626245683966408394965837152256.0; /* 0xffff * 2^208 */

/* ======================= logging / helpers ======================= */
static void logf_(const char *fmt, ...) {
    SYSTEMTIME t; GetLocalTime(&t);
    va_list ap; va_start(ap, fmt);
    EnterCriticalSection(&logCs);
    printf("[%02d:%02d:%02d] ", t.wHour, t.wMinute, t.wSecond);
    vprintf(fmt, ap); printf("\n"); fflush(stdout);
    LeaveCriticalSection(&logCs);
    va_end(ap);
}

static const char *fmtn(double d, char *buf) {
    static const char *u[] = {"", "K", "M", "G", "T", "P", "E", "Z"};
    int i = 0;
    if (isinf(d)) { strcpy(buf, "inf"); return buf; }
    if (d < 1000) { sprintf(buf, d < 10 ? "%.4g" : "%.2f", d); return buf; }
    while (d >= 1000 && i < 7) { d /= 1000; i++; }
    sprintf(buf, "%.2f%s", d, u[i]); return buf;
}

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = tolower(c); if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}
static int fromhex(const char *h, uint8_t *out, int max) {
    int n = (int)strlen(h); if (n % 2 || n / 2 > max) return -1;
    for (int i = 0; i < n / 2; i++) {
        int a = hexval(h[2*i]), b = hexval(h[2*i+1]); if (a < 0 || b < 0) return -1;
        out[i] = (uint8_t)(a << 4 | b);
    }
    return n / 2;
}
static void tohex(const uint8_t *b, int n, char *out) {
    static const char *x = "0123456789abcdef";
    for (int i = 0; i < n; i++) { out[2*i] = x[b[i] >> 4]; out[2*i+1] = x[b[i] & 15]; }
    out[2*n] = 0;
}
static void rev(uint8_t *b, int n) { for (int i = 0; i < n / 2; i++) { uint8_t t = b[i]; b[i] = b[n-1-i]; b[n-1-i] = t; } }
static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static uint32_t bswap(uint32_t x) { return __builtin_bswap32(x); }

/* ======================= SHA-256 (scalar) ======================= */
static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static const uint32_t IV[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
static inline void compress(uint32_t s[8], const uint32_t in[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = in[i];
    for (int i = 16; i < 64; i++) {
        uint32_t x = w[i-15], y = w[i-2];
        w[i] = w[i-16] + (ROTR(x,7) ^ ROTR(x,18) ^ (x >> 3)) + w[i-7] + (ROTR(y,17) ^ ROTR(y,19) ^ (y >> 10));
    }
    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
        uint32_t t2 = (ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22)) + ((a & b) | (c & (a | b)));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

static void sha256(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint32_t s[8], w[16]; uint8_t blk[64];
    memcpy(s, IV, sizeof s);
    size_t i = 0;
    for (; i + 64 <= len; i += 64) { for (int k = 0; k < 16; k++) w[k] = be32(data + i + 4*k); compress(s, w); }
    size_t r = len - i; memset(blk, 0, 64); memcpy(blk, data + i, r); blk[r] = 0x80;
    if (r >= 56) { for (int k = 0; k < 16; k++) w[k] = be32(blk + 4*k); compress(s, w); memset(blk, 0, 64); }
    uint64_t bits = (uint64_t)len * 8;
    for (int k = 0; k < 8; k++) blk[63-k] = (uint8_t)(bits >> (8*k));
    for (int k = 0; k < 16; k++) w[k] = be32(blk + 4*k);
    compress(s, w);
    for (int k = 0; k < 8; k++) { out[4*k]=s[k]>>24; out[4*k+1]=s[k]>>16; out[4*k+2]=s[k]>>8; out[4*k+3]=(uint8_t)s[k]; }
}
static void sha256d(const uint8_t *d, size_t n, uint8_t out[32]) { uint8_t t[32]; sha256(d, n, t); sha256(t, 32, out); }

/* ---- per-job precompute ----
 * Header block 2 = [b0 b1 b2 nonce 0x80000000 0 ... 0 640]. Only word 3 (the nonce) varies, so
 * rounds 0-2, most of round 3, and message-schedule words W16, W17 (fully) and W18, W19 (partly)
 * are the same for every nonce: compute them once per work unit instead of per hash. */
typedef struct { uint32_t mid[8], st[8], p3, t23, w16, w17, w18c, w19c; } Pre;

#define BS0(a) (ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22))
#define BS1(e) (ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25))
#define SS0(x) (ROTR(x,7) ^ ROTR(x,18) ^ ((x) >> 3))
#define SS1(x) (ROTR(x,17) ^ ROTR(x,19) ^ ((x) >> 10))
#define CHF(e,f,g) (((e) & (f)) ^ (~(e) & (g)))
#define MAJF(a,b,c) (((a) & (b)) | ((c) & ((a) | (b))))

static void make_pre(const uint32_t mid[8], uint32_t b0, uint32_t b1, uint32_t b2, Pre *p) {
    const uint32_t w[3] = {b0, b1, b2};
    uint32_t a=mid[0],b=mid[1],c=mid[2],d=mid[3],e=mid[4],f=mid[5],g=mid[6],h=mid[7];
    memcpy(p->mid, mid, 32);
    for (int i = 0; i < 3; i++) {
        uint32_t t1 = h + BS1(e) + CHF(e,f,g) + K[i] + w[i], t2 = BS0(a) + MAJF(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    p->st[0]=a; p->st[1]=b; p->st[2]=c; p->st[3]=d; p->st[4]=e; p->st[5]=f; p->st[6]=g; p->st[7]=h;
    p->p3  = h + BS1(e) + CHF(e,f,g) + K[3];      /* round 3: t1 = p3 + nonce word */
    p->t23 = BS0(a) + MAJF(a,b,c);                /* round 3: t2 (nonce-independent) */
    p->w16 = SS0(b1) + b0;                        /* s1(W14)=0, W9=0 */
    p->w17 = SS1(640u) + SS0(b2) + b1;            /* W10=0 */
    p->w18c = SS1(p->w16) + b2;                   /* W18 = w18c + s0(W3) */
    p->w19c = SS1(p->w17) + SS0(0x80000000u);     /* W19 = w19c + W3 */
}

/* Full SHA-256d of the header with nonce n, from the precomputed state. out = final state words. */
static inline void hash_pre(const Pre *p, uint32_t n, uint32_t out[8]) {
    uint32_t W[64], w3 = bswap(n);
    W[3] = w3; W[4] = 0x80000000u;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640; W[16] = p->w16; W[17] = p->w17; W[18] = p->w18c + SS0(w3); W[19] = p->w19c + w3;
    for (int i = 20; i < 64; i++) W[i] = SS1(W[i-2]) + W[i-7] + SS0(W[i-15]) + W[i-16];
    uint32_t a=p->st[0],b=p->st[1],c=p->st[2],d=p->st[3],e=p->st[4],f=p->st[5],g=p->st[6],h=p->st[7];
    uint32_t t1 = p->p3 + w3;
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+p->t23;
    for (int i = 4; i < 64; i++) {
        t1 = h + BS1(e) + CHF(e,f,g) + K[i] + W[i];
        uint32_t t2 = BS0(a) + MAJF(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    uint32_t w[16] = {p->mid[0]+a, p->mid[1]+b, p->mid[2]+c, p->mid[3]+d, p->mid[4]+e, p->mid[5]+f,
                      p->mid[6]+g, p->mid[7]+h, 0x80000000u, 0, 0, 0, 0, 0, 0, 256};
    memcpy(out, IV, 32);
    compress(out, w);
}

/* Scan nonces [n0, n0+count). Report nonces whose hash's top 32 bits (as LE number) <= limit. */
static int scan_scalar(const Pre *p, uint32_t n0, uint32_t count, uint32_t limit, uint32_t *out, int maxout) {
    int nf = 0; uint32_t s[8];
    for (uint32_t k = 0; k < count; k++) {
        uint32_t n = n0 + k;
        hash_pre(p, n, s);
        if (bswap(s[7]) <= limit && nf < maxout) out[nf++] = n;
    }
    return nf;
}

/* ======================= SHA-256 (AVX2, 8 lanes) ======================= */
#define VROTR(x,n) _mm256_or_si256(_mm256_srli_epi32(x,n), _mm256_slli_epi32(x,32-(n)))
#define VADD(a,b) _mm256_add_epi32(a,b)
#define VXOR(a,b) _mm256_xor_si256(a,b)

#define VS0(x) VXOR(VXOR(VROTR(x,7), VROTR(x,18)), _mm256_srli_epi32(x,3))
#define VS1(y) VXOR(VXOR(VROTR(y,17), VROTR(y,19)), _mm256_srli_epi32(y,10))
#define VROUND(i, Wi) do { \
        __m256i S1_ = VXOR(VXOR(VROTR(e,6), VROTR(e,11)), VROTR(e,25)); \
        __m256i ch_ = VXOR(_mm256_and_si256(e,f), _mm256_andnot_si256(e,g)); \
        __m256i t1_ = VADD(VADD(VADD(h, S1_), VADD(ch_, _mm256_set1_epi32((int)K[i]))), (Wi)); \
        __m256i S0_ = VXOR(VXOR(VROTR(a,2), VROTR(a,13)), VROTR(a,22)); \
        __m256i mj_ = _mm256_or_si256(_mm256_and_si256(a,b), _mm256_and_si256(c, _mm256_or_si256(a,b))); \
        h=g; g=f; f=e; e=VADD(d,t1_); d=c; c=b; b=a; a=VADD(t1_, VADD(S0_, mj_)); } while (0)

__attribute__((target("avx2"))) static inline void compress8(__m256i s[8], __m256i w[64]) {
    for (int i = 16; i < 64; i++) w[i] = VADD(VADD(w[i-16], VS0(w[i-15])), VADD(w[i-7], VS1(w[i-2])));
    __m256i a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    for (int i = 0; i < 64; i++) VROUND(i, w[i]);
    s[0]=VADD(s[0],a); s[1]=VADD(s[1],b); s[2]=VADD(s[2],c); s[3]=VADD(s[3],d);
    s[4]=VADD(s[4],e); s[5]=VADD(s[5],f); s[6]=VADD(s[6],g); s[7]=VADD(s[7],h);
}

__attribute__((target("avx2")))
static int scan_avx2(const Pre *p, uint32_t n0, uint32_t count, uint32_t limit, uint32_t *out, int maxout) {
    const __m256i bsw = _mm256_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,
                                         3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
    const __m256i zero = _mm256_setzero_si256(), pad = _mm256_set1_epi32((int)0x80000000);
    const __m256i L = _mm256_set1_epi32((int)limit), eight = _mm256_set1_epi32(8);
    __m256i M[8], I[8], ST[8];
    for (int i = 0; i < 8; i++) {
        M[i] = _mm256_set1_epi32((int)p->mid[i]); I[i] = _mm256_set1_epi32((int)IV[i]);
        ST[i] = _mm256_set1_epi32((int)p->st[i]);
    }
    const __m256i P3 = _mm256_set1_epi32((int)p->p3), T23 = _mm256_set1_epi32((int)p->t23);
    const __m256i W16 = _mm256_set1_epi32((int)p->w16), W17 = _mm256_set1_epi32((int)p->w17);
    const __m256i W18 = _mm256_set1_epi32((int)p->w18c), W19 = _mm256_set1_epi32((int)p->w19c);
    __m256i nonce = VADD(_mm256_set1_epi32((int)n0), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
    int nf = 0;
    __m256i w[64], s[8];
    for (uint32_t k = 0; k < count; k += 8) {
        /* first hash, block 2, starting at round 3 from the precomputed state */
        __m256i w3 = _mm256_shuffle_epi8(nonce, bsw);
        w[3] = w3; w[4] = pad;
        for (int i = 5; i < 15; i++) w[i] = zero;
        w[15] = _mm256_set1_epi32(640);
        w[16] = W16; w[17] = W17; w[18] = VADD(W18, VS0(w3)); w[19] = VADD(W19, w3);
        for (int i = 20; i < 64; i++) w[i] = VADD(VADD(w[i-16], VS0(w[i-15])), VADD(w[i-7], VS1(w[i-2])));
        __m256i a=ST[0],b=ST[1],c=ST[2],d=ST[3],e=ST[4],f=ST[5],g=ST[6],h=ST[7];
        __m256i t1 = VADD(P3, w3);
        h=g; g=f; f=e; e=VADD(d,t1); d=c; c=b; b=a; a=VADD(t1,T23);
        for (int i = 4; i < 64; i++) VROUND(i, w[i]);
        s[0]=VADD(M[0],a); s[1]=VADD(M[1],b); s[2]=VADD(M[2],c); s[3]=VADD(M[3],d);
        s[4]=VADD(M[4],e); s[5]=VADD(M[5],f); s[6]=VADD(M[6],g); s[7]=VADD(M[7],h);
        /* second hash */
        for (int i = 0; i < 8; i++) w[i] = s[i];
        w[8]=pad; for (int i = 9; i < 15; i++) w[i] = zero;
        w[15] = _mm256_set1_epi32(256);
        for (int i = 0; i < 8; i++) s[i] = I[i];
        compress8(s, w);
        __m256i top = _mm256_shuffle_epi8(s[7], bsw);
        __m256i le = _mm256_cmpeq_epi32(_mm256_min_epu32(top, L), top); /* top <= limit (unsigned) */
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(le));
        while (mask) {
            int lane = __builtin_ctz(mask); mask &= mask - 1;
            if (nf < maxout) out[nf++] = n0 + k + (uint32_t)lane;
        }
        nonce = VADD(nonce, eight);
    }
    return nf;
}

/* ======================= bitsliced SHA-256d, AVX2 (experiment, --bitslice) =======================
 * Transposed layout: a SHA word is 32 vectors, vector k = bit k of that word for 256 different nonces.
 * Rotations/shifts become free re-indexing; XOR/Ch/Maj are plain bitwise ops; additions become
 * explicit ripple-carry full adders (5 ops/bit, 2 ops/bit when one operand is a constant).
 * Lanes are 256 consecutive nonces (base multiple of 256): the nonce's low byte is the lane index,
 * which after the byte-swap lands in W3 bits 24..31 as fixed lane patterns - no runtime transpose. */
typedef __m256i BV;
typedef BV BW32[32];
static int useBitslice;

#define BX(a,b) _mm256_xor_si256(a,b)
#define BAND(a,b) _mm256_and_si256(a,b)
#define BOR(a,b) _mm256_or_si256(a,b)

__attribute__((target("avx2"))) static inline void bsv_add(BV *d, const BV *a, const BV *b) {
    BV x = a[0], y = b[0], c = BAND(x, y);
    d[0] = BX(x, y);
    for (int k = 1; k < 32; k++) {           /* reads a[k], b[k] before writing d[k]: d may alias a or b */
        x = a[k]; y = b[k];
        BV ax = BX(x, y);
        BV s = BX(ax, c);
        if (k < 31) c = BOR(BAND(x, y), BAND(c, ax));
        d[k] = s;
    }
}
__attribute__((target("avx2"))) static inline void bsv_addc(BV *d, const BV *a, uint32_t kc) {
    const BV ones = _mm256_set1_epi32(-1);
    BV c = _mm256_setzero_si256();
    for (int k = 0; k < 32; k++) {
        BV x = a[k];
        if ((kc >> k) & 1) { d[k] = BX(BX(x, c), ones); c = BOR(x, c); }
        else               { d[k] = BX(x, c);           c = BAND(x, c); }
    }
}
__attribute__((target("avx2"))) static inline void bsv_const(BV *d, uint32_t v) {
    const BV ones = _mm256_set1_epi32(-1), zero = _mm256_setzero_si256();
    for (int k = 0; k < 32; k++) d[k] = ((v >> k) & 1) ? ones : zero;
}
/* XOR of rotr(r1), rotr(r2), and rotr(r3) or shr(sh) (pass -1 for the unused one) */
__attribute__((target("avx2"))) static inline void bsv_sig(BV *d, const BV *x, int r1, int r2, int r3, int sh) {
    for (int k = 0; k < 32; k++) {
        BV v = BX(x[(k + r1) & 31], x[(k + r2) & 31]);
        if (r3 >= 0) v = BX(v, x[(k + r3) & 31]);
        if (sh >= 0 && k + sh < 32) v = BX(v, x[k + sh]);
        d[k] = v;
    }
}

/* one SHA-256 round on ring-buffered state (a = A[i&3], b = A[i-1], ..., h = E[i-3]): no state copying */
__attribute__((target("avx2")))
static inline void bsv_round(BW32 *A, BW32 *E, int i, const BV *W, uint32_t wconst, BV *t1, BV *t2, BV *t3, int needA) {
    BV *a = A[i & 3], *b = A[(i - 1) & 3], *c = A[(i - 2) & 3], *d = A[(i - 3) & 3];
    BV *e = E[i & 3], *f = E[(i - 1) & 3], *g = E[(i - 2) & 3], *h = E[(i - 3) & 3];
    bsv_sig(t2, e, 6, 11, 25, -1);
    bsv_add(t1, h, t2);
    for (int k = 0; k < 32; k++) t2[k] = BX(g[k], BAND(e[k], BX(f[k], g[k])));          /* Ch */
    bsv_add(t1, t1, t2);
    if (W) { bsv_add(t1, t1, W); bsv_addc(t1, t1, K[i]); } else bsv_addc(t1, t1, K[i] + wconst);
    bsv_add(E[(i + 1) & 3], d, t1);                                                      /* overwrites h */
    if (needA) {
        bsv_sig(t2, a, 2, 13, 22, -1);
        for (int k = 0; k < 32; k++) t3[k] = BOR(BAND(a[k], b[k]), BAND(c[k], BOR(a[k], b[k])));  /* Maj */
        bsv_add(t2, t2, t3);
        bsv_add(A[(i + 1) & 3], t1, t2);                                                 /* overwrites d */
    }
}

/* message schedule word i (rolling window of 16) with constant folding; the W[i-16] term goes
 * first because it lives in the destination slot */
__attribute__((target("avx2")))
static inline void bsv_expand(BW32 *Wb, int *isc, uint32_t *cv, int i, BV *t1, BV *t2) {
    int s = i & 15, i2 = (i - 2) & 15, i7 = (i - 7) & 15, i15 = (i - 15) & 15;
    uint32_t cacc = 0; const BV *v[4]; int nv = 0;
    if (isc[s]) cacc += cv[s]; else v[nv++] = Wb[s];
    if (isc[i2]) cacc += SS1(cv[i2]); else { bsv_sig(t1, Wb[i2], 17, 19, -1, 10); v[nv++] = t1; }
    if (isc[i7]) cacc += cv[i7]; else v[nv++] = Wb[i7];
    if (isc[i15]) cacc += SS0(cv[i15]); else { bsv_sig(t2, Wb[i15], 7, 18, -1, 3); v[nv++] = t2; }
    if (!nv) { isc[s] = 1; cv[s] = cacc; return; }
    if (nv == 1) bsv_addc(Wb[s], v[0], cacc);
    else {
        bsv_add(Wb[s], v[0], v[1]);
        for (int j = 2; j < nv; j++) bsv_add(Wb[s], Wb[s], v[j]);
        if (cacc) bsv_addc(Wb[s], Wb[s], cacc);
    }
    isc[s] = 0;
}

__attribute__((target("avx2")))
static int scan_bs_avx2(const Pre *p, uint32_t n0, uint32_t count, uint32_t limit, uint32_t *out, int maxout) {
    static __thread BV *mem;
    if (!mem) mem = _aligned_malloc(sizeof(BW32) * 28, 64);
    BW32 *A = (BW32 *)mem, *E = A + 4, *Wb = A + 8;
    BV *t1 = A[24], *t2 = A[25], *t3 = A[26], *W3 = A[27];
    BV lane[8];                                   /* lane pattern: bit b of lane index j */
    for (int b = 0; b < 8; b++) {
        uint64_t q[4];
        for (int e = 0; e < 4; e++) { q[e] = 0; for (int l = 0; l < 64; l++) if (((64 * e + l) >> b) & 1) q[e] |= 1ULL << l; }
        lane[b] = _mm256_loadu_si256((const __m256i *)q);
    }
    int lz = limit ? __builtin_clz(limit) : 32;   /* top lz bits of bswap(H7) must be zero */
    int nf = 0; int isc[16]; uint32_t cv[16];
    const uint64_t end = (uint64_t)n0 + count;
    for (uint64_t base = n0 & ~255u; base < end; base += 256) {
        uint32_t bb = bswap((uint32_t)base);
        bsv_const(W3, bb);
        for (int b = 0; b < 8; b++) W3[24 + b] = lane[b];

        /* ---- first hash, header block 2, from round 3 (rounds 0-2 precomputed) ---- */
        bsv_const(A[3], p->st[0]); bsv_const(A[2], p->st[1]); bsv_const(A[1], p->st[2]);
        bsv_const(E[3], p->st[4]); bsv_const(E[2], p->st[5]); bsv_const(E[1], p->st[6]);
        bsv_addc(t1, W3, p->p3);                  /* round 3: t1 = p3 + W3 */
        bsv_addc(E[0], t1, p->st[3]);             /* e = d + t1 */
        bsv_addc(A[0], t1, p->t23);               /* a = t1 + t2 */
        for (int k = 0; k < 16; k++) { isc[k] = 1; cv[k] = 0; }
        isc[3] = 0; memcpy(Wb[3], W3, sizeof(BW32));
        cv[4] = 0x80000000u; cv[15] = 640;
        for (int i = 4; i < 16; i++) bsv_round(A, E, i, NULL, cv[i], t1, t2, t3, 1);
        cv[0] = p->w16; cv[1] = p->w17;           /* W16, W17: constants */
        bsv_sig(t2, Wb[3], 7, 18, -1, 3); bsv_addc(Wb[2], t2, p->w18c); isc[2] = 0;   /* W18 */
        bsv_addc(Wb[3], Wb[3], p->w19c);          /* W19 = w19c + W3 */
        for (int i = 16; i < 64; i++) {
            if (i >= 20) bsv_expand(Wb, isc, cv, i, t2, t3);
            bsv_round(A, E, i, isc[i & 15] ? NULL : Wb[i & 15], cv[i & 15], t1, t2, t3, 1);
        }
        /* first hash = midstate + state (a=A0 b=A3 c=A2 d=A1 e=E0 f=E3 g=E2 h=E1) -> second hash input */
        const int ai[4] = {0, 3, 2, 1};
        for (int j = 0; j < 4; j++) { bsv_addc(Wb[j], A[ai[j]], p->mid[j]); bsv_addc(Wb[4 + j], E[ai[j]], p->mid[4 + j]); }
        for (int j = 0; j < 8; j++) isc[j] = 0;
        for (int j = 8; j < 16; j++) { isc[j] = 1; cv[j] = 0; }
        cv[8] = 0x80000000u; cv[15] = 256;

        /* ---- second hash: only H7 matters, so stop after round 60 (h_final = e from round 60) ---- */
        for (int j = 0; j < 4; j++) { bsv_const(A[ai[j]], IV[j]); bsv_const(E[ai[j]], IV[4 + j]); }
        for (int i = 0; i <= 60; i++) {
            if (i >= 16) bsv_expand(Wb, isc, cv, i, t2, t3);
            bsv_round(A, E, i, isc[i & 15] ? NULL : Wb[i & 15], cv[i & 15], t1, t2, t3, i < 60);
        }
        bsv_addc(t1, E[1], IV[7]);                /* H7 */

        /* bswap(H7) bit p comes from H7 bit 8*(3 - p/8) + p%8 */
        BV acc = _mm256_setzero_si256();
        for (int q = 31; q >= 32 - lz; q--) acc = BOR(acc, t1[8 * (3 - q / 8) + q % 8]);
        uint64_t m[4]; _mm256_storeu_si256((__m256i *)m, acc);
        for (int e = 0; e < 4; e++) {
            uint64_t z = ~m[e];
            while (z) {
                int l = __builtin_ctzll(z); z &= z - 1;
                uint64_t n = base + 64 * e + l;
                if (n >= n0 && n < end && nf < maxout) out[nf++] = (uint32_t)n;
            }
        }
    }
    return nf;
}

static int scan(const Pre *p, uint32_t n0, uint32_t count, uint32_t limit, uint32_t *out, int maxout) {
    if (useBitslice && useAvx2) return scan_bs_avx2(p, n0, count, limit, out, maxout);
    return useAvx2 ? scan_avx2(p, n0, count, limit, out, maxout) : scan_scalar(p, n0, count, limit, out, maxout);
}

/* ======================= GPU (OpenCL, loaded at runtime) ======================= */
typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_bitfield;
typedef struct _cl_platform_id *cl_platform_id; typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context; typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem; typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel; typedef struct _cl_event *cl_event;
#define CLAPI __stdcall
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_MEM_READ_WRITE 1

static cl_int (CLAPI *p_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
static cl_int (CLAPI *p_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void *, size_t *);
static cl_int (CLAPI *p_clGetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
static cl_int (CLAPI *p_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
static cl_context (CLAPI *p_clCreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
static cl_command_queue (CLAPI *p_clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int *);
static cl_program (CLAPI *p_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
static cl_int (CLAPI *p_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
static cl_int (CLAPI *p_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
static cl_kernel (CLAPI *p_clCreateKernel)(cl_program, const char *, cl_int *);
static cl_mem (CLAPI *p_clCreateBuffer)(cl_context, cl_bitfield, size_t, void *, cl_int *);
static cl_int (CLAPI *p_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
static cl_int (CLAPI *p_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
static cl_int (CLAPI *p_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
static cl_int (CLAPI *p_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
static cl_int (CLAPI *p_clFlush)(cl_command_queue);

/* One work-item = one nonce. Fully unrolled so the compiler folds constants, drops dead final rounds,
 * and maps rotate()/bitselect() onto Turing's funnel-shift and LOP3 instructions. */
static const char *KSRC =
"#define ROTR(x,n) rotate((uint)(x),(uint)(32-(n)))\n"
"#define BSWAP(x) as_uint(as_uchar4(x).s3210)\n"
"#define CH(e,f,g) bitselect((g),(f),(e))\n"
"#define MAJ(a,b,c) bitselect((a),(b),(a)^(c))\n"
"#define S0(a) (ROTR(a,2)^ROTR(a,13)^ROTR(a,22))\n"
"#define S1(e) (ROTR(e,6)^ROTR(e,11)^ROTR(e,25))\n"
"#define s0(x) (ROTR(x,7)^ROTR(x,18)^((x)>>3))\n"
"#define s1(x) (ROTR(x,17)^ROTR(x,19)^((x)>>10))\n"
"__constant uint K[64]={\n"
"0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,\n"
"0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,\n"
"0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,\n"
"0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,\n"
"0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,\n"
"0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,\n"
"0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,\n"
"0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};\n"
"inline void compress(uint *s, uint *w) {\n"
"  uint a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];\n"
"  #pragma unroll\n"
"  for (int i = 0; i < 64; i++) {\n"
"    if (i >= 16) w[i&15] += s1(w[(i-2)&15]) + w[(i-7)&15] + s0(w[(i-15)&15]);\n"
"    uint t1 = h + S1(e) + CH(e,f,g) + K[i] + w[i&15];\n"
"    uint t2 = S0(a) + MAJ(a,b,c);\n"
"    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;\n"
"  }\n"
"  s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;\n"
"}\n"
"/* st* = state after rounds 0-2 of header block 2, m* = midstate (for the final add),\n"
"   p3/t23 = nonce-independent parts of round 3, w16..w19c = precomputed message schedule. */\n"
"__kernel void search(uint st0, uint st1, uint st2, uint st3, uint st4, uint st5, uint st6, uint st7,\n"
"                     uint m0, uint m1, uint m2, uint m3, uint m4, uint m5, uint m6, uint m7,\n"
"                     uint p3, uint t23, uint w16, uint w17, uint w18c, uint w19c,\n"
"                     uint n0, uint limit, __global uint *out) {\n"
"  uint nonce = n0 + (uint)get_global_id(0);\n"
"  uint w3 = BSWAP(nonce);\n"
"  uint W[64];\n"
"  W[3] = w3; W[4] = 0x80000000u;\n"
"  #pragma unroll\n"
"  for (int i = 5; i < 15; i++) W[i] = 0;\n"
"  W[15] = 640u; W[16] = w16; W[17] = w17; W[18] = w18c + s0(w3); W[19] = w19c + w3;\n"
"  #pragma unroll\n"
"  for (int i = 20; i < 64; i++) W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];\n"
"  uint a=st0,b=st1,c=st2,d=st3,e=st4,f=st5,g=st6,h=st7;\n"
"  uint t1 = p3 + w3;\n"
"  h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t23;\n"
"  #pragma unroll\n"
"  for (int i = 4; i < 64; i++) {\n"
"    t1 = h + S1(e) + CH(e,f,g) + K[i] + W[i];\n"
"    uint t2 = S0(a) + MAJ(a,b,c);\n"
"    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;\n"
"  }\n"
"  uint w2[16] = {m0+a,m1+b,m2+c,m3+d,m4+e,m5+f,m6+g,m7+h,0x80000000u,0,0,0,0,0,0,256u};\n"
"  uint t[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};\n"
"  compress(t, w2);\n"
"  if (BSWAP(t[7]) <= limit) { uint i = atomic_inc(&out[0]); if (i < 255u) out[1+i] = nonce; }\n"
"}\n";

/* Bitsliced variant (--bitslice): one work-item = 32 consecutive nonces, one bit per lane of a uint.
 * Same algorithm as scan_bs_avx2. Full adder: sum = x^y^c (one LOP3), carry = maj = bitselect (one LOP3). */
static const char *KSRC_BS =
"__constant uint K[64]={\n"
"0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,\n"
"0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,\n"
"0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,\n"
"0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,\n"
"0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,\n"
"0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,\n"
"0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,\n"
"0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};\n"
"__constant uint IVc[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};\n"
"#define BSWAP(x) as_uint(as_uchar4(x).s3210)\n"
"#define RR(x,n) rotate((uint)(x),(uint)(32-(n)))\n"
"#define CS0(x) (RR(x,7)^RR(x,18)^((x)>>3))\n"
"#define CS1(x) (RR(x,17)^RR(x,19)^((x)>>10))\n"
"inline void bs_add(uint *d, const uint *a, const uint *b) {\n"
"  uint x = a[0], y = b[0], c = x & y; d[0] = x ^ y;\n"
"  #pragma unroll\n"
"  for (int k = 1; k < 32; k++) { x = a[k]; y = b[k]; uint s = x ^ y ^ c; c = bitselect(x, y, x ^ c); d[k] = s; }\n"
"}\n"
"inline void bs_addc(uint *d, const uint *a, uint kc) {\n"
"  uint c = 0;\n"
"  #pragma unroll\n"
"  for (int k = 0; k < 32; k++) { uint x = a[k], y = 0u - ((kc >> k) & 1u); uint s = x ^ y ^ c; c = bitselect(x, y, x ^ c); d[k] = s; }\n"
"}\n"
"inline void bs_const(uint *d, uint v) {\n"
"  #pragma unroll\n"
"  for (int k = 0; k < 32; k++) d[k] = 0u - ((v >> k) & 1u);\n"
"}\n"
"inline void bs_sig(uint *d, const uint *x, int r1, int r2, int r3, int sh) {\n"
"  #pragma unroll\n"
"  for (int k = 0; k < 32; k++) {\n"
"    uint v = x[(k + r1) & 31] ^ x[(k + r2) & 31];\n"
"    if (r3 >= 0) v ^= x[(k + r3) & 31];\n"
"    if (sh >= 0 && k + sh < 32) v ^= x[k + sh];\n"
"    d[k] = v;\n"
"  }\n"
"}\n"
"inline void bs_round(uint A[4][32], uint E[4][32], int i, const uint *W, int wc, uint wconst, uint *t1, uint *t2, uint *t3, int needA) {\n"
"  uint *a = A[i & 3], *b = A[(i - 1) & 3], *c = A[(i - 2) & 3], *d = A[(i - 3) & 3];\n"
"  uint *e = E[i & 3], *f = E[(i - 1) & 3], *g = E[(i - 2) & 3], *h = E[(i - 3) & 3];\n"
"  bs_sig(t2, e, 6, 11, 25, -1); bs_add(t1, h, t2);\n"
"  #pragma unroll\n"
"  for (int k = 0; k < 32; k++) t2[k] = bitselect(g[k], f[k], e[k]);\n"
"  bs_add(t1, t1, t2);\n"
"  if (!wc) { bs_add(t1, t1, W); bs_addc(t1, t1, K[i]); } else bs_addc(t1, t1, K[i] + wconst);\n"
"  bs_add(E[(i + 1) & 3], d, t1);\n"
"  if (needA) {\n"
"    bs_sig(t2, a, 2, 13, 22, -1);\n"
"    #pragma unroll\n"
"    for (int k = 0; k < 32; k++) t3[k] = bitselect(a[k], b[k], a[k] ^ c[k]);\n"
"    bs_add(t2, t2, t3); bs_add(A[(i + 1) & 3], t1, t2);\n"
"  }\n"
"}\n"
"inline void bs_expand(uint Wb[16][32], int *isc, uint *cv, int i, uint *t1, uint *t2) {\n"
"  int s = i & 15, i2 = (i - 2) & 15, i7 = (i - 7) & 15, i15 = (i - 15) & 15;\n"
"  uint cacc = 0; int nv = 0; int vs[4];\n"
"  /* vs codes: 0..15 = Wb slot, 16 = t1, 17 = t2 */\n"
"  if (isc[s]) cacc += cv[s]; else vs[nv++] = s;\n"
"  if (isc[i2]) cacc += CS1(cv[i2]); else { bs_sig(t1, Wb[i2], 17, 19, -1, 10); vs[nv++] = 16; }\n"
"  if (isc[i7]) cacc += cv[i7]; else vs[nv++] = i7;\n"
"  if (isc[i15]) cacc += CS0(cv[i15]); else { bs_sig(t2, Wb[i15], 7, 18, -1, 3); vs[nv++] = 17; }\n"
"  if (!nv) { isc[s] = 1; cv[s] = cacc; return; }\n"
"  for (int j = 0; j < nv; j++) {\n"
"    uint *v = vs[j] == 16 ? t1 : vs[j] == 17 ? t2 : Wb[vs[j]];\n"
"    if (j == 0) { if (nv == 1) bs_addc(Wb[s], v, cacc); else continue; }\n"
"    else if (j == 1) { uint *v0 = vs[0] == 16 ? t1 : vs[0] == 17 ? t2 : Wb[vs[0]]; bs_add(Wb[s], v0, v); }\n"
"    else bs_add(Wb[s], Wb[s], v);\n"
"  }\n"
"  if (nv > 1 && cacc) bs_addc(Wb[s], Wb[s], cacc);\n"
"  isc[s] = 0;\n"
"}\n"
"__kernel void search_bs(uint st0, uint st1, uint st2, uint st3, uint st4, uint st5, uint st6, uint st7,\n"
"                        uint m0, uint m1, uint m2, uint m3, uint m4, uint m5, uint m6, uint m7,\n"
"                        uint p3, uint t23, uint w16, uint w17, uint w18c, uint w19c,\n"
"                        uint n0, uint limit, __global uint *out) {\n"
"  uint base = n0 + (uint)get_global_id(0) * 32u;\n"
"  uint A[4][32], E[4][32], Wb[16][32], t1[32], t2[32], t3[32];\n"
"  int isc[16]; uint cv[16];\n"
"  const uint mid[8] = {m0,m1,m2,m3,m4,m5,m6,m7};\n"
"  const uint lane[5] = {0xAAAAAAAAu, 0xCCCCCCCCu, 0xF0F0F0F0u, 0xFF00FF00u, 0xFFFF0000u};\n"
"  uint bb = BSWAP(base);\n"
"  bs_const(Wb[3], bb);\n"
"  for (int b = 0; b < 5; b++) Wb[3][24 + b] = lane[b];\n"
"  bs_const(A[3], st0); bs_const(A[2], st1); bs_const(A[1], st2);\n"
"  bs_const(E[3], st4); bs_const(E[2], st5); bs_const(E[1], st6);\n"
"  bs_addc(t1, Wb[3], p3); bs_addc(E[0], t1, st3); bs_addc(A[0], t1, t23);\n"
"  for (int k = 0; k < 16; k++) { isc[k] = 1; cv[k] = 0; }\n"
"  isc[3] = 0; cv[4] = 0x80000000u; cv[15] = 640u;\n"
"  for (int i = 4; i < 16; i++) bs_round(A, E, i, t3, 1, cv[i], t1, t2, t3, 1);\n"
"  cv[0] = w16; cv[1] = w17;\n"
"  bs_sig(t2, Wb[3], 7, 18, -1, 3); bs_addc(Wb[2], t2, w18c); isc[2] = 0;\n"
"  bs_addc(Wb[3], Wb[3], w19c);\n"
"  for (int i = 16; i < 64; i++) {\n"
"    if (i >= 20) bs_expand(Wb, isc, cv, i, t2, t3);\n"
"    bs_round(A, E, i, Wb[i & 15], isc[i & 15], cv[i & 15], t1, t2, t3, 1);\n"
"  }\n"
"  const int ai[4] = {0, 3, 2, 1};\n"
"  for (int j = 0; j < 4; j++) { bs_addc(Wb[j], A[ai[j]], mid[j]); bs_addc(Wb[4 + j], E[ai[j]], mid[4 + j]); }\n"
"  for (int j = 0; j < 8; j++) isc[j] = 0;\n"
"  for (int j = 8; j < 16; j++) { isc[j] = 1; cv[j] = 0; }\n"
"  cv[8] = 0x80000000u; cv[15] = 256u;\n"
"  for (int j = 0; j < 4; j++) { bs_const(A[ai[j]], IVc[j]); bs_const(E[ai[j]], IVc[4 + j]); }\n"
"  for (int i = 0; i <= 60; i++) {\n"
"    if (i >= 16) bs_expand(Wb, isc, cv, i, t2, t3);\n"
"    bs_round(A, E, i, Wb[i & 15], isc[i & 15], cv[i & 15], t1, t2, t3, i < 60);\n"
"  }\n"
"  bs_addc(t1, E[1], IVc[7]);\n"
"  int lz = limit ? clz(limit) : 32;\n"
"  uint acc = 0;\n"
"  for (int q = 31; q >= 32 - lz; q--) acc |= t1[8 * (3 - q / 8) + q % 8];\n"
"  uint z = ~acc;\n"
"  while (z) {\n"
"    int l = 31 - clz(z & (0u - z)); z &= z - 1u;\n"
"    uint idx = atomic_inc(&out[0]); if (idx < 255u) out[1 + idx] = base + (uint)l;\n"
"  }\n"
"}\n";

/* Version rolling (BIP 310, "overt AsicBoost"): NVER header versions differ only in header block 1, so they
 * share header block 2 - and therefore the whole first-hash message schedule W18..W63 - for every nonce.
 * Each work-item expands W once, then runs NVER midstates through it. NVER (1/2/4/8) is a build option.
 * Per-version data comes from a small __constant buffer: VD[v*18 + 0..7] = state after rounds 0-2,
 * [8..15] = midstate (for the final add), [16] = p3, [17] = t23. Appended to KSRC (uses its macros). */
static const char *KSRC_VR =
"__kernel void search_vrn(__constant uint *VD, uint4 WC, uint n0, uint limit, __global uint *out) {\n"
"  uint nonce = n0 + (uint)get_global_id(0);\n"
"  uint w3 = BSWAP(nonce);\n"
"  uint W[64];\n"
"  W[3] = w3; W[4] = 0x80000000u;\n"
"  #pragma unroll\n"
"  for (int i = 5; i < 15; i++) W[i] = 0;\n"
"  W[15] = 640u; W[16] = WC.s0; W[17] = WC.s1; W[18] = WC.s2 + s0(w3); W[19] = WC.s3 + w3;\n"
"  #pragma unroll\n"
"  for (int i = 20; i < 64; i++) W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];   /* shared by all versions */\n"
"  #pragma unroll\n"
"  for (int v = 0; v < NVER; v++) {\n"
"    __constant uint *X = VD + v * 18;\n"
"    uint a=X[0],b=X[1],c=X[2],d=X[3],e=X[4],f=X[5],g=X[6],h=X[7];\n"
"    uint t1 = X[16] + w3; h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+X[17];\n"
"    #pragma unroll\n"
"    for (int i = 4; i < 64; i++) {\n"
"      t1 = h + S1(e) + CH(e,f,g) + K[i] + W[i]; uint t2 = S0(a) + MAJ(a,b,c);\n"
"      h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;\n"
"    }\n"
"#ifndef SECOND_PRE\n"
"    uint w2[16] = {X[8]+a,X[9]+b,X[10]+c,X[11]+d,X[12]+e,X[13]+f,X[14]+g,X[15]+h,0x80000000u,0,0,0,0,0,0,256u};\n"
"    uint t[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};\n"
"    compress(t, w2);\n"
"    uint H7 = t[7];\n"
"#else\n"
"    /* explicit second hash: round 0 starts from the constant IV, so everything but W0 is folded into\n"
"       SP_A / SP_E (host-computed); and only H7 matters, which is IV7 + e after round 60 -> stop there */\n"
"    uint V[61];\n"
"    V[0]=X[8]+a; V[1]=X[9]+b; V[2]=X[10]+c; V[3]=X[11]+d; V[4]=X[12]+e; V[5]=X[13]+f; V[6]=X[14]+g; V[7]=X[15]+h;\n"
"    V[8]=0x80000000u; V[9]=0; V[10]=0; V[11]=0; V[12]=0; V[13]=0; V[14]=0; V[15]=256u;\n"
"    #pragma unroll\n"
"    for (int i = 16; i < 61; i++) V[i] = s1(V[i-2]) + V[i-7] + s0(V[i-15]) + V[i-16];\n"
"    uint A = SP_A + V[0], B = 0x6a09e667u, C = 0xbb67ae85u, D = 0x3c6ef372u;\n"
"    uint E = SP_E + V[0], F = 0x510e527fu, G = 0x9b05688cu, H = 0x1f83d9abu;\n"
"    #pragma unroll\n"
"    for (int i = 1; i <= 60; i++) {\n"
"      uint u1 = H + S1(E) + CH(E,F,G) + K[i] + V[i], u2 = S0(A) + MAJ(A,B,C);\n"
"      H=G; G=F; F=E; E=D+u1; D=C; C=B; B=A; A=u1+u2;\n"
"    }\n"
"    uint H7 = 0x5be0cd19u + E;\n"
"#endif\n"
"    if (BSWAP(H7) <= limit) { uint q = atomic_inc(&out[0]); if (q < 127u) { out[1+2*q] = nonce; out[2+2*q] = (uint)v; } }\n"
"  }\n"
"}\n";

#define VR_REQ_MASK 0x1fffe000u   /* BIP 320 general-purpose version bits */
#define VR_MAX 16
static int vrCount = 8;            /* --versions 1|2|4|8|16 (8 measured best: +13.0% vs +10.9% for 4) */
static int vrLog2 = 3;
static int secondPre = 1;          /* explicit second-hash round-0 precompute + stop after round 60 (+0.3%, on by default) */
static int gpuMaxRegs;             /* --gpu-maxregs N: NVIDIA register cap for the version-rolling kernel */
static size_t gpuLocal;            /* --gpu-local N: fixed workgroup size (0 = driver chooses) */
static char vrRegInfo[160];        /* registers / spill / occupancy of the version-rolling kernel */
static int freshBuild;             /* bench/self-test: bypass the driver's kernel cache so the build log (registers) is produced */
/* The NVIDIA driver caches compiled modules by their PTX, so an unused -D does not force a rebuild. This tag
 * kernel puts the build stamp into the PTX itself (only when PCSOLO_FRESH is defined); it is never launched. */
static const char *KSRC_TAG =
"#ifdef PCSOLO_FRESH\n"
"__kernel void pcsolo_fresh_tag(__global uint *o) { o[0] = (uint)(PCSOLO_FRESH); }\n"
"#endif\n";
static int wantVR;                 /* --version-roll */
static volatile int vrOn;          /* pool agreed (or offline bench/self-test) */
static volatile uint32_t vrMask;

static int useGpu, gpuIntensity = 22, gpuDevice = -1;
static volatile LONG64 gpuHashes;
static char gpuName[128];
static cl_command_queue gq; static cl_kernel gk, gkbs, gkvr; static cl_mem gout[4]; /* one per in-flight launch (GPU_DEPTH) */
static cl_mem gvd[4];   /* per-launch version data for the version-rolling kernel */

/* NVIDIA's -cl-nv-verbose build log: "Compiling entry function 'NAME'", "... N bytes spill stores", "Used N registers" */
static void parse_ptxas(const char *log, const char *fn, int *regs, int *spill) {
    char key[96]; snprintf(key, sizeof key, "entry function '%s'", fn);
    const char *p = strstr(log, key), *q;
    *regs = -1; *spill = -1;
    if (!p) return;
    if ((q = strstr(p, "stack frame, "))) sscanf(q, "stack frame, %d bytes spill stores", spill);
    if ((q = strstr(p, "Used "))) sscanf(q, "Used %d registers", regs);
}
/* Turing SM: 64K registers, max 32 warps; registers are allocated per warp in units of 256 */
static int warps_per_sm(int regs) {
    if (regs <= 0) return -1;
    int perWarp = ((regs * 32 + 255) / 256) * 256, w = 65536 / perWarp;
    return w > 32 ? 32 : w;
}

static int gpu_init(void) {
    HMODULE h = LoadLibraryA("OpenCL.dll");
    if (!h) { printf("GPU: OpenCL.dll not found (no GPU driver with OpenCL?)\n"); return -1; }
#define LOADCL(f) if (!(*(FARPROC *)&p_##f = GetProcAddress(h, #f))) { printf("GPU: missing " #f "\n"); return -1; }
    LOADCL(clGetPlatformIDs) LOADCL(clGetPlatformInfo) LOADCL(clGetDeviceIDs) LOADCL(clGetDeviceInfo)
    LOADCL(clCreateContext) LOADCL(clCreateCommandQueue) LOADCL(clCreateProgramWithSource) LOADCL(clBuildProgram)
    LOADCL(clGetProgramBuildInfo) LOADCL(clCreateKernel) LOADCL(clCreateBuffer) LOADCL(clSetKernelArg)
    LOADCL(clEnqueueNDRangeKernel) LOADCL(clEnqueueReadBuffer) LOADCL(clEnqueueWriteBuffer) LOADCL(clFlush)

    cl_platform_id plats[8]; cl_uint np = 0;
    if (p_clGetPlatformIDs(8, plats, &np) || !np) { printf("GPU: no OpenCL platforms\n"); return -1; }
    cl_device_id devs[16]; int nd = 0, pick = -1;
    for (cl_uint p = 0; p < np && nd < 16; p++) {
        cl_uint n = 0; char pname[128] = "";
        p_clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof pname, pname, NULL);
        if (p_clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 16 - nd, devs + nd, &n)) continue;
        for (cl_uint i = 0; i < n; i++, nd++) {
            char dn[128] = ""; p_clGetDeviceInfo(devs[nd], CL_DEVICE_NAME, sizeof dn, dn, NULL);
            printf("GPU %d: %s (%s)\n", nd, dn, pname);
            if (pick < 0 && strstr(pname, "NVIDIA")) pick = nd;
        }
    }
    if (!nd) { printf("GPU: no OpenCL GPU devices\n"); return -1; }
    if (gpuDevice >= 0) pick = gpuDevice;
    if (pick < 0) pick = 0;
    if (pick >= nd) { printf("GPU: device %d does not exist\n", pick); return -1; }
    cl_device_id dev = devs[pick];
    p_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof gpuName, gpuName, NULL);

    cl_int e;
    cl_context ctx = p_clCreateContext(NULL, 1, &dev, NULL, NULL, &e); if (e) { printf("GPU: context error %d\n", e); return -1; }
    gq = p_clCreateCommandQueue(ctx, dev, 0, &e); if (e) { printf("GPU: queue error %d\n", e); return -1; }
    char ext[4096] = ""; p_clGetDeviceInfo(dev, 0x1030 /* CL_DEVICE_EXTENSIONS */, sizeof ext, ext, NULL);
    const int nvVerbose = strstr(ext, "cl_nv_compiler_options") != NULL;
    char nocache[48] = "";
    if (freshBuild) snprintf(nocache, sizeof nocache, " -D PCSOLO_FRESH=%uu", (unsigned)GetTickCount64());
    char bopts[96]; snprintf(bopts, sizeof bopts, "%s%s", nvVerbose ? "-cl-nv-verbose" : "", nocache);
    const char *msrcs[2] = {KSRC, KSRC_TAG};
    cl_program prog = p_clCreateProgramWithSource(ctx, 2, msrcs, NULL, &e); if (e) { printf("GPU: program error %d\n", e); return -1; }
    if ((e = p_clBuildProgram(prog, 1, &dev, bopts, NULL, NULL))) {
        static char logb[16384]; logb[0] = 0;
        p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof logb - 1, logb, NULL);
        printf("GPU: kernel build failed (%d):\n%s\n", e, logb); return -1;
    }
    gk = p_clCreateKernel(prog, "search", &e); if (e) { printf("GPU: kernel error %d\n", e); return -1; }
    int baseRegs = -1, baseSpill = -1;
    if (nvVerbose) { static char lg[32768]; lg[0] = 0; p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof lg - 1, lg, NULL); parse_ptxas(lg, "search", &baseRegs, &baseSpill); }
    if (wantVR) {   /* version-rolling kernel for the chosen version count */
        const char *srcs[3] = {KSRC, KSRC_VR, KSRC_TAG};
        char extra[160] = "";
        int xn = 0;
        if (secondPre) {   /* second hash round 0 from the constant IV: t1 = C0 + W0, t2 = T0 */
            uint32_t C0 = IV[7] + BS1(IV[4]) + CHF(IV[4], IV[5], IV[6]) + K[0], T0 = BS0(IV[0]) + MAJF(IV[0], IV[1], IV[2]);
            xn += snprintf(extra + xn, sizeof extra - xn, " -D SECOND_PRE -D SP_A=0x%08xu -D SP_E=0x%08xu", C0 + T0, IV[3] + C0);
        }
        if (gpuMaxRegs > 0 && nvVerbose) xn += snprintf(extra + xn, sizeof extra - xn, " -cl-nv-maxrregcount=%d", gpuMaxRegs);
        char opts[320]; snprintf(opts, sizeof opts, "-D NVER=%d%s%s%s", vrCount, nvVerbose ? " -cl-nv-verbose" : "", nocache, extra);
        cl_program pvr = p_clCreateProgramWithSource(ctx, 3, srcs, NULL, &e); if (e) { printf("GPU: program error %d\n", e); return -1; }
        static char lg[65536]; lg[0] = 0;
        e = p_clBuildProgram(pvr, 1, &dev, opts, NULL, NULL);
        p_clGetProgramBuildInfo(pvr, dev, CL_PROGRAM_BUILD_LOG, sizeof lg - 1, lg, NULL);
        if (e) { printf("GPU: version-rolling kernel build failed (%d):\n%s\n", e, lg); return -1; }
        gkvr = p_clCreateKernel(pvr, "search_vrn", &e); if (e) { printf("GPU: kernel error %d\n", e); return -1; }
        for (int i = 0; i < 4; i++) {
            gvd[i] = p_clCreateBuffer(ctx, 4 /* CL_MEM_READ_ONLY */, VR_MAX * 18 * 4, NULL, &e);
            if (e) { printf("GPU: buffer error %d\n", e); return -1; }
        }
        if (getenv("PCSOLO_KLOG")) printf("---- version-rolling kernel build log (%s) ----\n%s\n----\n", opts, lg);
        int r = -1, s = -1; if (nvVerbose) parse_ptxas(lg, "search_vrn", &r, &s);
        char basePart[64] = "";   /* the normal kernel's log is empty when the driver serves it from its cache */
        if (baseRegs > 0) snprintf(basePart, sizeof basePart, " (normal kernel: %d registers, %d/32 warps)", baseRegs, warps_per_sm(baseRegs));
        if (r > 0)
            snprintf(vrRegInfo, sizeof vrRegInfo, "x%d versions: %d registers, %d B spill, %d/32 warps per SM%s",
                     vrCount, r, s, warps_per_sm(r), basePart);
        else snprintf(vrRegInfo, sizeof vrRegInfo, "x%d versions (register usage not reported by this driver)", vrCount);
    }
    if (useBitslice) {
        cl_program pbs = p_clCreateProgramWithSource(ctx, 1, &KSRC_BS, NULL, &e); if (e) { printf("GPU: program error %d\n", e); return -1; }
        if ((e = p_clBuildProgram(pbs, 1, &dev, "", NULL, NULL))) {
            static char logb[16384]; logb[0] = 0;
            p_clGetProgramBuildInfo(pbs, dev, CL_PROGRAM_BUILD_LOG, sizeof logb - 1, logb, NULL);
            printf("GPU: bitsliced kernel build failed (%d):\n%s\n", e, logb); return -1;
        }
        gkbs = p_clCreateKernel(pbs, "search_bs", &e); if (e) { printf("GPU: kernel error %d\n", e); return -1; }
    }
    for (int i = 0; i < 4; i++) {
        gout[i] = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, 256 * 4, NULL, &e);
        if (e) { printf("GPU: buffer error %d\n", e); return -1; }
    }
    return 0;
}

/* Queue a scan of nonces [n0, n0+count) into result buffer `buf` (0/1). Does not wait. */
static int gpu_launch(int buf, const Pre *p, uint32_t n0, size_t count, uint32_t limit) {
    static const uint32_t zero = 0;
    const uint32_t extra[6] = {p->p3, p->t23, p->w16, p->w17, p->w18c, p->w19c};
    cl_kernel k = useBitslice ? gkbs : gk;
    size_t items = useBitslice ? count / 32 : count;   /* bitsliced: 32 nonces per work-item */
    cl_int e = p_clEnqueueWriteBuffer(gq, gout[buf], 0, 0, 4, &zero, 0, NULL, NULL);
    for (int i = 0; i < 8; i++) e |= p_clSetKernelArg(k, i, 4, &p->st[i]);
    for (int i = 0; i < 8; i++) e |= p_clSetKernelArg(k, 8 + i, 4, &p->mid[i]);
    for (int i = 0; i < 6; i++) e |= p_clSetKernelArg(k, 16 + i, 4, &extra[i]);
    e |= p_clSetKernelArg(k, 22, 4, &n0); e |= p_clSetKernelArg(k, 23, 4, &limit);
    e |= p_clSetKernelArg(k, 24, sizeof(cl_mem), &gout[buf]);
    e |= p_clEnqueueNDRangeKernel(gq, k, 1, NULL, &items, NULL, 0, NULL, NULL); /* args are copied at enqueue */
    e |= p_clFlush(gq);
    return e ? -1 : 0;
}

/* Wait for the launch that wrote `buf` and fetch its candidates. Returns count, or -1 on error. */
static int gpu_collect(int buf, uint32_t *out, int maxout) {
    uint32_t res[256];
    if (p_clEnqueueReadBuffer(gq, gout[buf], 1, 0, sizeof res, res, 0, NULL, NULL)) return -1;
    int nf = res[0] < 255 ? (int)res[0] : 255;
    if (nf > maxout) nf = maxout;
    memcpy(out, res + 1, nf * 4);
    return nf;
}

/* Synchronous helper (self-test). */
static int gpu_scan(const Pre *p, uint32_t n0, size_t count, uint32_t limit, uint32_t *out, int maxout) {
    if (gpu_launch(0, p, n0, count, limit)) return -1;
    return gpu_collect(0, out, maxout);
}

/* pack per-version kernel data: [v*18 + 0..7] state after rounds 0-2, [8..15] midstate, [16] p3, [17] t23 */
static void pack_vd(const Pre *pv, int n, uint32_t *vd) {
    for (int v = 0; v < n; v++) {
        memcpy(vd + v * 18, pv[v].st, 32); memcpy(vd + v * 18 + 8, pv[v].mid, 32);
        vd[v * 18 + 16] = pv[v].p3; vd[v * 18 + 17] = pv[v].t23;
    }
}

/* Version-rolling launch: vrCount versions (w16..w19c are identical across them). vd must stay
 * untouched until the launch completes (it lives in the in-flight Work slot) - the write is non-blocking. */
static int gpu_launch_vr(int buf, const uint32_t *vd, const Pre *p0, uint32_t n0, size_t count, uint32_t limit) {
    static const uint32_t zero = 0;
    const uint32_t WC[4] = {p0->w16, p0->w17, p0->w18c, p0->w19c};
    cl_int e = p_clEnqueueWriteBuffer(gq, gout[buf], 0, 0, 4, &zero, 0, NULL, NULL);
    e |= p_clEnqueueWriteBuffer(gq, gvd[buf], 0, 0, (size_t)vrCount * 18 * 4, vd, 0, NULL, NULL);
    e |= p_clSetKernelArg(gkvr, 0, sizeof(cl_mem), &gvd[buf]);
    e |= p_clSetKernelArg(gkvr, 1, sizeof WC, WC);
    e |= p_clSetKernelArg(gkvr, 2, 4, &n0); e |= p_clSetKernelArg(gkvr, 3, 4, &limit);
    e |= p_clSetKernelArg(gkvr, 4, sizeof(cl_mem), &gout[buf]);
    e |= p_clEnqueueNDRangeKernel(gq, gkvr, 1, NULL, &count, (gpuLocal && count % gpuLocal == 0) ? &gpuLocal : NULL, 0, NULL, NULL);
    e |= p_clFlush(gq);
    return e ? -1 : 0;
}

/* Collect (nonce, version index) pairs. Returns pair count, or -1 on error. */
static int gpu_collect_vr(int buf, uint32_t *nonces, uint32_t *vis, int maxout) {
    uint32_t res[256];
    if (p_clEnqueueReadBuffer(gq, gout[buf], 1, 0, sizeof res, res, 0, NULL, NULL)) return -1;
    int nf = res[0] < 127 ? (int)res[0] : 127;
    if (nf > maxout) nf = maxout;
    for (int i = 0; i < nf; i++) { nonces[i] = res[1 + 2 * i]; vis[i] = res[2 + 2 * i] & (VR_MAX - 1); }
    return nf;
}

/* ======================= difficulty helpers ======================= */
static double diff_of(const uint8_t le[32]) {
    double v = 0; for (int i = 31; i >= 0; i--) v = v * 256 + le[i];
    return v == 0 ? INFINITY : DIFF1 / v;
}
static void target_from_bits(uint32_t bits, uint8_t t[32]) {
    memset(t, 0, 32); int e = (int)(bits >> 24); uint32_t m = bits & 0xffffff;
    for (int i = 0; i < 3; i++) { int idx = e - 3 + i; if (idx >= 0 && idx < 32) t[idx] = (uint8_t)(m >> (8*i)); }
}
static int le_leq(const uint8_t a[32], const uint8_t b[32]) {
    for (int i = 31; i >= 0; i--) { if (a[i] < b[i]) return 1; if (a[i] > b[i]) return 0; }
    return 1;
}
static uint32_t top_limit(double d) {
    if (d >= 1) return 0;
    double l = floor(1.0 / d); return l >= 4294967295.0 ? 0xffffffffu : (uint32_t)l;
}

/* ======================= minimal JSON ======================= */
enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ };
typedef struct J { int t; double n; char *s; struct J **v; char **k; int len; } J;

static void jfree(J *j) {
    if (!j) return;
    for (int i = 0; i < j->len; i++) { jfree(j->v[i]); if (j->k) free(j->k[i]); }
    free(j->v); free(j->k); free(j->s); free(j);
}
static void jws(const char **p) { while (**p == ' ' || **p == '\t' || **p == '\r' || **p == '\n') (*p)++; }
static char *jstr(const char **p) {
    (*p)++; size_t cap = 64, n = 0; char *s = malloc(cap);
    while (**p && **p != '"') {
        char c = *(*p)++;
        if (c == '\\') {
            c = *(*p)++;
            if (c == 'n') c = '\n'; else if (c == 't') c = '\t'; else if (c == 'r') c = '\r';
            else if (c == 'u') { for (int i = 0; i < 4 && **p; i++) (*p)++; c = '?'; }
        }
        if (n + 2 > cap) s = realloc(s, cap *= 2);
        s[n++] = c;
    }
    if (**p == '"') (*p)++;
    s[n] = 0; return s;
}
static J *jparse(const char **p) {
    jws(p); J *j = calloc(1, sizeof *j);
    char c = **p;
    if (c == '{' || c == '[') {
        int obj = c == '{'; j->t = obj ? JOBJ : JARR; (*p)++;
        int cap = 0;
        for (;;) {
            jws(p);
            if (**p == (obj ? '}' : ']')) { (*p)++; break; }
            if (!**p) break;
            char *key = NULL;
            if (obj) { if (**p != '"') break; key = jstr(p); jws(p); if (**p == ':') (*p)++; }
            if (j->len == cap) { cap = cap ? cap * 2 : 8; j->v = realloc(j->v, cap * sizeof *j->v); if (obj) j->k = realloc(j->k, cap * sizeof *j->k); }
            if (obj) j->k[j->len] = key;
            j->v[j->len++] = jparse(p);
            jws(p); if (**p == ',') (*p)++;
        }
    } else if (c == '"') { j->t = JSTR; j->s = jstr(p); }
    else if (!strncmp(*p, "true", 4)) { j->t = JBOOL; j->n = 1; *p += 4; }
    else if (!strncmp(*p, "false", 5)) { j->t = JBOOL; *p += 5; }
    else if (!strncmp(*p, "null", 4)) { j->t = JNULL; *p += 4; }
    else { char *e; j->t = JNUM; j->n = strtod(*p, &e); if (e == *p) (*p)++; else *p = e; }
    return j;
}
static J *jget(J *o, const char *key) {
    if (!o || o->t != JOBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (!strcmp(o->k[i], key)) return o->v[i];
    return NULL;
}
static J *jat(J *a, int i) { return (a && a->t == JARR && i < a->len) ? a->v[i] : NULL; }
static const char *js(J *j) { return (j && j->t == JSTR) ? j->s : ""; }

/* ======================= stratum ======================= */
static void send_line(const char *s) {
    EnterCriticalSection(&sendCs);
    SOCKET so = sock;
    if (so != INVALID_SOCKET) { send(so, s, (int)strlen(s), 0); send(so, "\n", 1, 0); }
    LeaveCriticalSection(&sendCs);
}

static const char *err_text(J *e, char *buf) {
    if (e && e->t == JARR && e->len > 1 && e->v[1]->t == JSTR) return e->v[1]->s;
    strcpy(buf, e && e->t != JNULL ? "error" : "unknown"); return buf;
}

static void on_notify(J *p) {
    if (!en1ok || !p || p->t != JARR || p->len < 8) return;
    static Job j; /* only touched by the network thread */
    memset(&j, 0, sizeof j);
    snprintf(j.id, sizeof j.id, "%s", js(jat(p,0)));
    if (fromhex(js(jat(p,1)), j.prev, 32) != 32) return;
    for (int i = 0; i < 32; i += 4) rev(j.prev + i, 4);
    if ((j.cb1len = fromhex(js(jat(p,2)), j.cb1, MAXCB)) < 0) { logf_("coinbase too large, skipping job"); return; }
    if ((j.cb2len = fromhex(js(jat(p,3)), j.cb2, MAXCB)) < 0) { logf_("coinbase too large, skipping job"); return; }
    J *br = jat(p,4);
    j.nbr = br && br->t == JARR ? br->len : 0;
    if (j.nbr > MAXBR) return;
    for (int i = 0; i < j.nbr; i++) if (fromhex(js(br->v[i]), j.br[i], 32) != 32) return;
    if (fromhex(js(jat(p,5)), j.ver4, 4) != 4) return;
    rev(j.ver4, 4);
    if (fromhex(js(jat(p,6)), j.nbits4, 4) != 4) return;
    rev(j.nbits4, 4);
    snprintf(j.ntimehex, sizeof j.ntimehex, "%s", js(jat(p,7)));
    if (fromhex(j.ntimehex, j.ntime4, 4) != 4) return;
    rev(j.ntime4, 4);
    target_from_bits(strtoul(js(jat(p,6)), NULL, 16), j.target);
    j.netdiff = diff_of(j.target);
    memcpy(j.en1, en1, en1len); j.en1len = en1len; j.en2size = en2size;

    AcquireSRWLockExclusive(&jobLock);
    int newBlock = !haveJob || memcmp(gjob.prev, j.prev, 32) != 0;
    gjob = j; jobGen++; haveJob = 1;
    ReleaseSRWLockExclusive(&jobLock);
    char b[32];
    if (newBlock) logf_("New block template | network difficulty %s", fmtn(j.netdiff, b));
}

static void log_share_result(long long id, int ok, const char *err);

static int handle(const char *line) {
    const char *p = line; J *m = jparse(&p); int rc = 0; char eb[32], b[32];
    J *method = jget(m, "method");
    if (method && method->t == JSTR) {
        J *pa = jget(m, "params");
        if (!strcmp(method->s, "mining.notify")) on_notify(pa);
        else if (!strcmp(method->s, "mining.set_difficulty") && jat(pa,0)) {
            poolDiff = jat(pa,0)->n; logf_("Pool share difficulty: %s", fmtn(poolDiff, b));
        } else if (!strcmp(method->s, "mining.set_extranonce") && jat(pa,1)) {
            en1len = fromhex(js(jat(pa,0)), en1, 32); en2size = (int)jat(pa,1)->n;
        } else if (!strcmp(method->s, "client.reconnect")) { logf_("Pool asked us to reconnect"); rc = -1; }
        else if (!strcmp(method->s, "mining.set_version_mask") && jat(pa,0)) {
            vrMask = (uint32_t)strtoul(js(jat(pa,0)), NULL, 16) & VR_REQ_MASK;
            vrOn = wantVR && __builtin_popcount(vrMask) >= vrLog2;
            logf_("Pool version mask: %08x (version rolling %s)", vrMask, vrOn ? "on" : "off");
        }
    } else {
        J *idj = jget(m, "id"), *res = jget(m, "result"), *err = jget(m, "error");
        int id = idj && idj->t == JNUM ? (int)idj->n : 0;
        if (id == 4) {                                    /* mining.configure (BIP 310) */
            J *ok = jget(res, "version-rolling"), *mk = jget(res, "version-rolling.mask");
            if (ok && ok->t == JBOOL && ok->n && mk && mk->t == JSTR) {
                vrMask = (uint32_t)strtoul(mk->s, NULL, 16) & VR_REQ_MASK;
                vrOn = __builtin_popcount(vrMask) >= vrLog2;
                logf_("Version rolling %s (mask %08x)", vrOn ? "ON" : "unusable", vrMask);
            } else logf_("Pool does not support version rolling - GPU uses the normal kernel");
        } else if (id == 1) {
            if (!res || res->t != JARR || res->len < 3) { logf_("Subscribe failed: %s", err_text(err, eb)); rc = -1; }
            else {
                en1len = fromhex(js(res->v[1]), en1, 32); en2size = (int)res->v[2]->n; en1ok = en1len >= 0;
                logf_("Subscribed (extranonce1 %s, extranonce2 %d bytes)", js(res->v[1]), en2size);
            }
        } else if (id == 2) {
            if (res && res->t == JBOOL && res->n) logf_("Authorized as %s", user);
            else { logf_("Authorization FAILED: %s (check your address)", err_text(err, eb)); exit(2); }
        } else if (id >= 10) {
            int ok = res && res->t == JBOOL && res->n;
            InterlockedIncrement(ok ? &accepted : &rejected);
            log_share_result(idj->n, ok, ok ? "" : err_text(err, eb));
        }
    }
    jfree(m); return rc;
}

/* Share bookkeeping: each submission remembers where it came from, so the pool's answer can be logged
 * per version. Index VR_MAX in the tallies = shares without version rolling (CPU, normal GPU kernel). */
typedef struct { long long id; int src, vi; uint32_t vbits, nonce; double diff; } ShareMeta;
static ShareMeta smeta[512];
static volatile LONG accV[VR_MAX + 1], rejV[VR_MAX + 1], badCand;

/* vr: append BIP 310 version bits (the rolled version masked with the negotiated mask) */
static void submit(const Job *j, const char *en2hex, uint32_t nonce, int vr, uint32_t vbits, int src, int vi, double diff) {
    char buf[512], ver[24] = "";
    long long id = (long long)InterlockedIncrement64(&submitId);
    ShareMeta *m = &smeta[id % 512];              /* stored before sending, so the reply always finds it */
    m->src = src; m->vi = vr ? vi : -1; m->vbits = vbits; m->nonce = nonce; m->diff = diff; m->id = id;
    if (vr) snprintf(ver, sizeof ver, ",\"%08x\"", vbits);
    snprintf(buf, sizeof buf,
        "{\"id\":%lld,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%08x\"%s]}",
        id, user, j->id, en2hex, j->ntimehex, nonce, ver);
    send_line(buf);
}

static void log_share_result(long long id, int ok, const char *err) {
    ShareMeta m = smeta[id % 512];
    char d[32], tag[48];
    if (m.id != id) { logf_("share #%lld %s", id, ok ? "accepted" : "REJECTED"); return; }
    if (m.vi >= 0) snprintf(tag, sizeof tag, "%s v%d [bits %08x]", m.src ? "GPU" : "CPU", m.vi, m.vbits);
    else snprintf(tag, sizeof tag, "%s", m.src ? "GPU" : "CPU");
    int slot = m.vi >= 0 ? m.vi : VR_MAX;
    InterlockedIncrement(ok ? &accV[slot] : &rejV[slot]);
    logf_("share #%lld %s nonce %08x diff %s (hash re-verified) -> %s%s%s", id, tag, m.nonce, fmtn(m.diff, d),
          ok ? "accepted" : "REJECTED", ok ? "" : ": ", ok ? "" : err);
}

static void session(void) {
    struct addrinfo hints = {0}, *ai = NULL;
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    logf_("Connecting to %s:%s ...", host, portstr);
    if (getaddrinfo(host, portstr, &hints, &ai) != 0) { logf_("DNS lookup failed"); return; }
    SOCKET s = INVALID_SOCKET;
    for (struct addrinfo *a = ai; a; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == INVALID_SOCKET) continue;
        if (connect(s, a->ai_addr, (int)a->ai_addrlen) == 0) break;
        closesocket(s); s = INVALID_SOCKET;
    }
    freeaddrinfo(ai);
    if (s == INVALID_SOCKET) { logf_("Could not connect"); return; }
    DWORD to = 300000; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof to);
    BOOL nd = TRUE; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&nd, sizeof nd);
    sock = s; en1ok = 0; vrOn = 0;

    char buf[512];
    if (wantVR)   /* BIP 310: ask for version rolling before subscribing */
        send_line("{\"id\":4,\"method\":\"mining.configure\",\"params\":[[\"version-rolling\"],"
                  "{\"version-rolling.mask\":\"1fffe000\",\"version-rolling.min-bit-count\":2}]}");
    send_line("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"" VERSION "\"]}");
    snprintf(buf, sizeof buf, "{\"id\":3,\"method\":\"mining.suggest_difficulty\",\"params\":[%.10g]}", suggestDiff);
    send_line(buf);
    snprintf(buf, sizeof buf, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}", user);
    send_line(buf);

    static char rb[1 << 16]; int used = 0;
    for (;;) {
        if (used >= (int)sizeof rb - 1) { logf_("Line too long from pool"); break; }
        int r = recv(s, rb + used, (int)sizeof rb - 1 - used, 0);
        if (r <= 0) { logf_(r == 0 ? "Pool closed the connection" : "Connection error/timeout"); break; }
        used += r; rb[used] = 0;
        char *start = rb, *nl; int stop = 0;
        while ((nl = strchr(start, '\n'))) {
            *nl = 0;
            if (nl > start && handle(start) < 0) { stop = 1; break; }
            start = nl + 1;
        }
        if (stop) break;
        used = (int)(rb + used - start); memmove(rb, start, used);
    }
    haveJob = 0;
    EnterCriticalSection(&sendCs); sock = INVALID_SOCKET; LeaveCriticalSection(&sendCs);
    closesocket(s);
}

/* ======================= mining ======================= */
/* src: 0 = CPU, 1 = GPU; vi = version index (-1 = no version rolling). The candidate's double SHA-256 is
 * recomputed here from the exact header (with its rolled version) - this is what gets submitted. */
static void on_candidate(const Job *j, uint8_t hdr[80], uint32_t n, const char *en2hex, double pd,
                         int vr, uint32_t vbits, int vi, int src) {
    uint8_t h[32]; char b[32], hx[65];
    if (offline) return;                  /* synthetic bench/calibration job */
    hdr[76] = (uint8_t)n; hdr[77] = (uint8_t)(n >> 8); hdr[78] = (uint8_t)(n >> 16); hdr[79] = (uint8_t)(n >> 24);
    sha256d(hdr, 80, h);
    double d = diff_of(h);
    /* the kernel only reports hashes whose top 32 bits pass the filter; anything else means the reported
     * (nonce, version) does not reproduce - a kernel/host mismatch. Count it and never submit it. */
    uint32_t top = (uint32_t)h[31] << 24 | (uint32_t)h[30] << 16 | (uint32_t)h[29] << 8 | h[28];
    if (top > top_limit(pd)) {
        InterlockedIncrement(&badCand);
        logf_("WARNING: %s candidate nonce %08x v%d failed exact re-verification (top %08x)", src ? "GPU" : "CPU", n, vi, top);
        return;
    }
    EnterCriticalSection(&bestCs);
    if (d > bestDiff) { bestDiff = d; if (d >= 1) logf_("New best difficulty: %s", fmtn(d, b)); }
    LeaveCriticalSection(&bestCs);
    int block = le_leq(h, j->target);
    if (block) {
        InterlockedIncrement(&blocksFound);
        rev(h, 32); tohex(h, 32, hx);
        logf_("!!!!!!!! BLOCK FOUND !!!!!!!! hash %s", hx);
    }
    if (block || d >= pd) submit(j, en2hex, n, vr, vbits, src, vi, d);
}

/* Snapshot the current job and build fresh work: new extranonce2 -> new merkle root -> new 2^32 nonce space. */
static LONG prep_work(Job *j, uint8_t *cb, uint8_t hdr[80], Pre *pre, char en2hex[40]) {
    uint8_t root[32], cat[64]; uint32_t w[16], mid[8];
    AcquireSRWLockShared(&jobLock);
    *j = gjob; LONG gen = jobGen;
    ReleaseSRWLockShared(&jobLock);

    uint64_t e2 = (uint64_t)InterlockedIncrement64(&en2Counter);
    uint8_t en2[16] = {0}; int es = j->en2size > 16 ? 16 : j->en2size;
    for (int i = 0; i < es && i < 8; i++) en2[es - 1 - i] = (uint8_t)(e2 >> (8 * i));
    tohex(en2, es, en2hex);
    int n = 0;
    memcpy(cb + n, j->cb1, j->cb1len); n += j->cb1len;
    memcpy(cb + n, j->en1, j->en1len); n += j->en1len;
    memcpy(cb + n, en2, es); n += es;
    memcpy(cb + n, j->cb2, j->cb2len); n += j->cb2len;
    sha256d(cb, n, root);
    for (int i = 0; i < j->nbr; i++) { memcpy(cat, root, 32); memcpy(cat + 32, j->br[i], 32); sha256d(cat, 64, root); }

    memcpy(hdr, j->ver4, 4); memcpy(hdr + 4, j->prev, 32); memcpy(hdr + 36, root, 32);
    memcpy(hdr + 68, j->ntime4, 4); memcpy(hdr + 72, j->nbits4, 4); memset(hdr + 76, 0, 4);
    memcpy(mid, IV, 32);
    for (int i = 0; i < 16; i++) w[i] = be32(hdr + 4 * i);
    compress(mid, w);
    make_pre(mid, be32(hdr + 64), be32(hdr + 68), be32(hdr + 72), pre);
    return gen;
}

static DWORD WINAPI miner(LPVOID arg) {
    const int idx = (int)(intptr_t)arg;   /* threads with idx >= activeThreads stay parked */
    Job *j = malloc(sizeof *j);
    uint8_t hdr[80], *cb = malloc(2 * MAXCB + 64);
    uint32_t found[256];
    Pre pre;
    char en2hex[40];
    for (;;) {
        if (!haveJob || idx >= activeThreads) { Sleep(100); continue; }
        LONG gen = prep_work(j, cb, hdr, &pre, en2hex);
        uint32_t n0 = 0;
        do {
            double pd = poolDiff;
            int nf = scan(&pre, n0, CHUNK, top_limit(pd), found, 256);
            for (int i = 0; i < nf; i++) on_candidate(j, hdr, found[i], en2hex, pd, 0, 0, -1, 0);
            InterlockedAdd64(&totalHashes, CHUNK);
            n0 += CHUNK;
        } while (n0 != 0 && haveJob && jobGen == gen && idx < activeThreads);
    }
    return 0;
}

/* GPU feeder with a launch queue GPU_DEPTH deep: we only wait for the oldest launch while the
 * newer ones are already queued, so the GPU keeps working while the host reads results, builds
 * new work, or is briefly descheduled (the process runs at idle priority). Each launch remembers
 * its work slot, because its results may belong to an older work unit. */
#define GPU_DEPTH 4
typedef struct {
    Job job; uint8_t hdr[80]; char en2hex[40]; Pre pre;
    int vr; uint8_t hv[VR_MAX][80]; Pre pv[VR_MAX]; uint32_t vbits[VR_MAX];   /* version-rolling variants */
    uint32_t vd[VR_MAX * 18];                                                  /* packed for the kernel */
} Work;

/* put the bits of v into the lowest set bits of mask */
static uint32_t vr_deposit(uint32_t v, uint32_t mask) {
    uint32_t r = 0;
    for (int b = 0; mask && (v >> b); b++, mask &= mask - 1) if ((v >> b) & 1) r |= mask & (0u - mask);
    return r;
}

/* n header versions of one work unit: same block 2 (merkle tail, time, bits), different block 1 */
static void make_versions(const uint8_t hdr[80], uint32_t mask, int n, uint8_t hv[][80], Pre pv[], uint32_t vbits[]) {
    uint32_t jv = (uint32_t)hdr[0] | (uint32_t)hdr[1] << 8 | (uint32_t)hdr[2] << 16 | (uint32_t)hdr[3] << 24;
    for (int v = 0; v < n; v++) {
        uint32_t rv = (jv & ~mask) | vr_deposit((uint32_t)v, mask), mid[8], w[16];
        memcpy(hv[v], hdr, 80);
        hv[v][0] = (uint8_t)rv; hv[v][1] = (uint8_t)(rv >> 8); hv[v][2] = (uint8_t)(rv >> 16); hv[v][3] = (uint8_t)(rv >> 24);
        memcpy(mid, IV, 32);
        for (int i = 0; i < 16; i++) w[i] = be32(hv[v] + 4 * i);
        compress(mid, w);
        make_pre(mid, be32(hv[v] + 64), be32(hv[v] + 68), be32(hv[v] + 72), &pv[v]);
        vbits[v] = rv & mask;
    }
}
typedef struct { int slot; double pd; } Flight;

static DWORD WINAPI gpu_miner(LPVOID arg) {
    (void)arg;
    Work *wk[GPU_DEPTH + 1];
    for (int i = 0; i <= GPU_DEPTH; i++) wk[i] = malloc(sizeof(Work));
    uint8_t *cb = malloc(2 * MAXCB + 64);
    uint32_t found[256], vis[128];
    const size_t G = (size_t)1 << gpuIntensity;
    Flight fl[GPU_DEPTH];                 /* ring indexed by result buffer */
    int head = 0, inflight = 0, cur = -1, need = 1;
    double periodMs = 0;
    uint32_t n0 = 0; LONG gen = 0;
    LARGE_INTEGER qf, tPrev = {0}, tNow; QueryPerformanceFrequency(&qf);

    for (;;) {
        if (!haveJob) {                                   /* disconnected: drop in-flight results */
            while (inflight) { gpu_collect((head - inflight + GPU_DEPTH) % GPU_DEPTH, found, 255); inflight--; }
            need = 1; tPrev.QuadPart = 0; Sleep(200); continue;
        }
        if (need) {                                       /* pick a work slot no in-flight launch uses */
            for (int s = 0; s <= GPU_DEPTH; s++) {
                int used = 0;
                for (int k = 0; k < inflight; k++) if (fl[(head - 1 - k + GPU_DEPTH) % GPU_DEPTH].slot == s) used = 1;
                if (!used) { cur = s; break; }
            }
            gen = prep_work(&wk[cur]->job, cb, wk[cur]->hdr, &wk[cur]->pre, wk[cur]->en2hex);
            wk[cur]->vr = wantVR && vrOn;
            if (wk[cur]->vr) {
                make_versions(wk[cur]->hdr, vrMask, vrCount, wk[cur]->hv, wk[cur]->pv, wk[cur]->vbits);
                pack_vd(wk[cur]->pv, vrCount, wk[cur]->vd);
            }
            n0 = 0; need = 0;
        }
        double pd = poolDiff;
        /* version rolling hashes vrCount headers per nonce: launch 1/vrCount of the nonces so each launch
         * takes about as long as in normal mode (same desktop smoothness, same queue timing) */
        const size_t L = wk[cur]->vr ? (G >> vrLog2) : G;
        int lerr = wk[cur]->vr ? gpu_launch_vr(head, wk[cur]->vd, &wk[cur]->pv[0], n0, L, top_limit(pd))
                               : gpu_launch(head, &wk[cur]->pre, n0, L, top_limit(pd));
        if (lerr) { logf_("GPU error - GPU mining stopped (CPU continues)"); return 1; }
        fl[head].slot = cur; fl[head].pd = pd;
        head = (head + 1) % GPU_DEPTH; inflight++;
        LONG64 hashes = (LONG64)L * (wk[cur]->vr ? vrCount : 1);   /* each version is a distinct header */
        InterlockedAdd64(&totalHashes, hashes);
        InterlockedAdd64(&gpuHashes, hashes);
        n0 += (uint32_t)L;
        if (n0 == 0 || jobGen != gen) need = 1;

        if (inflight == GPU_DEPTH) {
            /* queue full: wait for the oldest. NVIDIA's blocking read spins a core, so sleep through
             * most of one launch period first - the newer launches keep the GPU busy meanwhile. */
            if (periodMs > 4) Sleep((DWORD)(periodMs * 0.8));
            int ob = (head - inflight + GPU_DEPTH) % GPU_DEPTH;
            Work *pw = wk[fl[ob].slot];
            int nf = pw->vr ? gpu_collect_vr(ob, found, vis, 127) : gpu_collect(ob, found, 255);
            if (nf < 0) { logf_("GPU error - GPU mining stopped (CPU continues)"); return 1; }
            inflight--;
            QueryPerformanceCounter(&tNow);
            if (tPrev.QuadPart) {
                double ms = (double)(tNow.QuadPart - tPrev.QuadPart) * 1000.0 / (double)qf.QuadPart;
                periodMs = periodMs ? 0.8 * periodMs + 0.2 * ms : ms;
            }
            tPrev = tNow;
            for (int i = 0; i < nf; i++) {
                if (pw->vr) on_candidate(&pw->job, pw->hv[vis[i]], found[i], pw->en2hex, fl[ob].pd, 1, pw->vbits[vis[i]], (int)vis[i], 1);
                else        on_candidate(&pw->job, pw->hdr, found[i], pw->en2hex, fl[ob].pd, 0, 0, -1, 1);
            }
        }
    }
    return 0;
}

static void start_miners(void) {
    static int started;
    if (started) return;
    started = 1;
    for (int i = 0; i < nthreads; i++) {
        HANDLE t = CreateThread(NULL, 0, miner, (LPVOID)(intptr_t)i, 0, NULL);
        SetThreadPriority(t, THREAD_PRIORITY_BELOW_NORMAL);
        CloseHandle(t);
    }
    if (useGpu) {
        HANDLE t = CreateThread(NULL, 0, gpu_miner, NULL, 0, NULL);
        SetThreadPriority(t, THREAD_PRIORITY_ABOVE_NORMAL); /* keep the GPU fed */
        CloseHandle(t);
    }
}

static DWORD WINAPI stats(LPVOID arg) {
    (void)arg;
    LONG64 last = 0, lastG = 0; ULONGLONG lt = GetTickCount64();
    for (;;) {
        Sleep(15000);
        LONG64 now = totalHashes, nowG = gpuHashes; ULONGLONG t = GetTickCount64();
        double secs = (double)(t - lt) / 1000.0;
        double rate = (double)(now - last) / secs, grate = (double)(nowG - lastG) / secs;
        last = now; lastG = nowG; lt = t;
        char a[32], b[32], c[32], d[32], e[32], odds[96] = "", split[96] = "";
        if (haveJob && rate > 0) {
            double years = gjob.netdiff * 4294967296.0 / rate / 31557600.0;
            snprintf(odds, sizeof odds, " | expected time to a block: %s years", fmtn(years, c));
        }
        if (useGpu) snprintf(split, sizeof split, " (CPU %sH/s + GPU %sH/s)", fmtn(rate - grate, d), fmtn(grate, e));
        logf_("%sH/s%s | shares %ld ok / %ld rejected | best diff %s%s",
              fmtn(rate, a), split, accepted, rejected, fmtn(bestDiff, b), odds);
    }
    return 0;
}

static BOOL WINAPI on_ctrl(DWORD type) {
    (void)type;
    double secs = (double)(GetTickCount64() - startTick) / 1000.0; char a[32], b[32], c[32];
    logf_("Stopped. %s hashes in %.0fs, avg %sH/s, shares %ld/%ld, best diff %s, blocks %ld",
          fmtn((double)totalHashes, a), secs, fmtn(totalHashes / (secs > 1 ? secs : 1), b),
          accepted, rejected, fmtn(bestDiff, c), blocksFound);
    char line[256] = ""; int n = 0;
    if (accV[VR_MAX] || rejV[VR_MAX]) n += snprintf(line + n, sizeof line - n, " unversioned %ld/%ld", accV[VR_MAX], rejV[VR_MAX]);
    for (int v = 0; v < VR_MAX; v++)
        if (accV[v] || rejV[v]) n += snprintf(line + n, sizeof line - n, " | v%d %ld/%ld", v, accV[v], rejV[v]);
    logf_("Shares accepted/rejected by source:%s | failed re-verification: %ld", n ? line : " none", badCand);
    return FALSE;
}

/* ======================= self-test / bench ======================= */
static int selftest(void) {
    static const char *gen = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c";
    static const char *expect = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";
    uint8_t hdr[80], h[32], t[32]; char hx[65], b[32]; int ok = 1;

    sha256((const uint8_t *)"abc", 3, h); tohex(h, 32, hx);
    int r = !strcmp(hx, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    printf("sha256(\"abc\")      : %s\n", r ? "OK" : hx); ok &= r;

    fromhex(gen, hdr, 80);
    sha256d(hdr, 80, h); rev(h, 32); tohex(h, 32, hx);
    r = !strcmp(hx, expect); printf("genesis SHA-256d   : %s\n", r ? "OK" : hx); ok &= r;

    uint32_t mid[8], w[16], found[8];
    memcpy(mid, IV, 32); for (int i = 0; i < 16; i++) w[i] = be32(hdr + 4 * i); compress(mid, w);
    uint32_t nonce = (uint32_t)hdr[76] | (uint32_t)hdr[77] << 8 | (uint32_t)hdr[78] << 16 | (uint32_t)hdr[79] << 24;
    Pre pre; make_pre(mid, be32(hdr+64), be32(hdr+68), be32(hdr+72), &pre);

    /* precomputed path vs reference SHA-256d, word for word, for 64 nonces */
    r = 1;
    for (uint32_t k = 0; k < 64; k++) {
        uint8_t hh[80], ref[32]; uint32_t s[8], n = nonce - 30 + k;
        memcpy(hh, hdr, 76); hh[76] = (uint8_t)n; hh[77] = (uint8_t)(n >> 8); hh[78] = (uint8_t)(n >> 16); hh[79] = (uint8_t)(n >> 24);
        sha256d(hh, 80, ref); hash_pre(&pre, n, s);
        for (int i = 0; i < 8; i++) if (s[i] != be32(ref + 4 * i)) r = 0;
    }
    printf("precomputed hash   : %s (64 nonces vs reference)\n", r ? "OK" : "FAIL"); ok &= r;

    int nf = scan_scalar(&pre, nonce - 21, 64, 0, found, 8);
    r = nf == 1 && found[0] == nonce; printf("scalar nonce scan  : %s\n", r ? "OK" : "FAIL"); ok &= r;

    /* cross-check every backend on 64K nonces with a loose filter (~1/256 pass) - candidate sets must match */
    static uint32_t ref[1024], cand[1024];
    int nref = scan_scalar(&pre, 0x12340000u, 65536, 0x00ffffffu, ref, 1024);
    if (useAvx2) {
        nf = scan_avx2(&pre, nonce - 21, 64, 0, found, 8);
        r = nf == 1 && found[0] == nonce;
        int nc = scan_avx2(&pre, 0x12340000u, 65536, 0x00ffffffu, cand, 1024);
        r &= nc == nref && !memcmp(cand, ref, nref * 4);
        printf("AVX2 nonce scan    : %s (%d/%d candidates match scalar)\n", r ? "OK" : "FAIL", nc, nref); ok &= r;
    } else printf("AVX2 nonce scan    : skipped (CPU has no AVX2)\n");
    if (useAvx2) {
        nf = scan_bs_avx2(&pre, nonce - 21, 64, 0, found, 8);
        r = nf == 1 && found[0] == nonce;
        int nc = scan_bs_avx2(&pre, 0x12340000u, 65536, 0x00ffffffu, cand, 1024);
        r &= nc == nref && !memcmp(cand, ref, nref * 4);
        printf("bitsliced AVX2     : %s (%d/%d candidates match scalar)\n", r ? "OK" : "FAIL", nc, nref); ok &= r;
    }
    if (useGpu) {
        /* 4M-nonce launch with the genesis nonce near the end: proves large launches cover every nonce */
        LARGE_INTEGER qf, t0, t1; QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&t0);
        uint32_t gfound[8];
        nf = gpu_scan(&pre, (nonce - 4000000u) & ~31u, 1u << 22, 0, gfound, 8);   /* aligned for the bitsliced kernel */
        QueryPerformanceCounter(&t1);
        r = 0; for (int i = 0; i < nf; i++) if (gfound[i] == nonce) r = 1;
        /* GPU reports candidates in arbitrary order: compare as sets (sorted) */
        int nc = gpu_scan(&pre, 0x12340000u, 65536, 0x00ffffffu, cand, 255);
        int setok = nc == (nref < 255 ? nref : 255);
        for (int i = 0; i < nc && setok; i++) { int hit = 0; for (int k = 0; k < nref; k++) if (cand[i] == ref[k]) hit = 1; setok &= hit; }
        r &= setok;
        printf("GPU %s scan   : %s (4M nonces in %.1f ms, %d/%d candidates match scalar)\n", useBitslice ? "bitsliced" : "nonce    ", r ? "OK" : "FAIL",
               (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)qf.QuadPart, nc, nref); ok &= r;

    }
    if (useGpu && wantVR) {
        /* version rolling: vrCount header versions sharing one message schedule per nonce */
        static uint8_t hv[VR_MAX][80]; static Pre pv[VR_MAX]; static uint32_t vb[VR_MAX], vd[VR_MAX * 18];
        uint32_t vn[128], vv[128];
        make_versions(hdr, VR_REQ_MASK, vrCount, hv, pv, vb);
        pack_vd(pv, vrCount, vd);
        const uint32_t span = 16384u / (uint32_t)vrCount;   /* keeps ~64 loose-filter candidates under the 127 cap */
        int pre_ok = 1;
        for (int v = 0; v < vrCount; v++)
            for (uint32_t k = 0; k < 4; k++) {        /* each version's precompute vs reference SHA-256d of the rolled header */
                uint8_t hh[80], rf[32]; uint32_t s[8], n = 0x12340000u + k * 977u;
                memcpy(hh, hv[v], 76); hh[76] = (uint8_t)n; hh[77] = (uint8_t)(n >> 8); hh[78] = (uint8_t)(n >> 16); hh[79] = (uint8_t)(n >> 24);
                sha256d(hh, 80, rf); hash_pre(&pv[v], n, s);
                for (int i = 0; i < 8; i++) if (s[i] != be32(rf + 4 * i)) pre_ok = 0;
            }
        int gen_ok = 0;                                /* genesis version is untouched at v0 -> nonce must be found there */
        if (!gpu_launch_vr(0, vd, &pv[0], nonce - 1000, 4096, 0)) {
            int ng = gpu_collect_vr(0, vn, vv, 127);
            for (int i = 0; i < ng; i++) if (vn[i] == nonce && vv[i] == 0) gen_ok = 1;
        }
        static uint32_t en_[1024], ev_[1024]; int nexp = 0;   /* loose filter: (nonce, version) sets must match */
        for (int v = 0; v < vrCount; v++) {
            int c = scan_scalar(&pv[v], 0x12340000u, span, 0x00ffffffu, cand, 1024);
            for (int i = 0; i < c && nexp < 1024; i++) { en_[nexp] = cand[i]; ev_[nexp++] = (uint32_t)v; }
        }
        int ng = -1, set_ok = 0;
        if (!gpu_launch_vr(0, vd, &pv[0], 0x12340000u, span, 0x00ffffffu)) ng = gpu_collect_vr(0, vn, vv, 127);
        if (ng == nexp && nexp < 127) {
            set_ok = 1;
            for (int i = 0; i < ng; i++) { int hit = 0; for (int k = 0; k < nexp; k++) if (vn[i] == en_[k] && vv[i] == ev_[k]) hit = 1; set_ok &= hit; }
        }
        r = pre_ok && gen_ok && set_ok;
        printf("GPU version-roll   : %s (x%d: precompute %s, genesis at v0 %s, %d/%d (nonce,version) candidates match scalar)\n",
               r ? "OK" : "FAIL", vrCount, pre_ok ? "ok" : "BAD", gen_ok ? "ok" : "MISSING", ng, nexp); ok &= r;
    }

    target_from_bits(0x1d00ffff, t); sha256d(hdr, 80, h);
    r = le_leq(h, t) && fabs(diff_of(t) - 1) < 1e-9; printf("target/difficulty  : %s\n", r ? "OK" : "FAIL"); ok &= r;
    printf("genesis block diff : %s\n", fmtn(diff_of(h), b));
    printf(ok ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    return ok ? 0 : 1;
}

/* offline synthetic job: same hashing paths as live mining, no pool, candidates ignored */
static void synthetic_job(void) {
    memset(&gjob, 0, sizeof gjob);
    strcpy(gjob.id, "bench"); gjob.cb1len = 40; gjob.cb2len = 40; gjob.en1len = 4; gjob.en2size = 4;
    gjob.ver4[3] = 0x20; gjob.netdiff = 1.4e14; strcpy(gjob.ntimehex, "00000000");
    offline = 1; poolDiff = 1e300;
    if (wantVR) { vrMask = VR_REQ_MASK; vrOn = 1; }
    InterlockedIncrement(&jobGen); haveJob = 1;
}

/* Auto-tune: measure total (CPU + GPU) throughput at several CPU thread counts, on the real hashing paths
 * with the GPU running, and keep the fewest threads within 1% of the best (less heat/power for ~same rate).
 * Coarse pass, then the neighbours of the winner. ~25 s. */
static int autotune(void) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    const int L = (int)si.dwNumberOfProcessors;
    double rate[129] = {0}, best = 0; int tested[129] = {0};
    char a[32], b[32], c[32];
    printf("Auto-tuning CPU threads%s%s (offline, ~25 s)...\n", useGpu ? " alongside the GPU" : "", wantVR ? " with version rolling" : "");
    synthetic_job();
    activeThreads = 0;
    start_miners();
    Sleep(1500);                                   /* GPU / clocks settle */
    int cand[16], nc = 0;
    int coarse[5] = {L / 2, (3 * L) / 4, L - 2, L - 1, L};
    for (int i = 0; i < 5; i++) if (coarse[i] >= 1 && coarse[i] <= L && coarse[i] <= 128) cand[nc++] = coarse[i];
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < nc; i++) {
            int n = cand[i];
            if (tested[n]) continue;
            tested[n] = 1;
            activeThreads = n;
            Sleep(1000);                           /* threads unpark at the next 64K-nonce chunk */
            LONG64 h0 = totalHashes, g0 = gpuHashes; ULONGLONG t0 = GetTickCount64();
            Sleep(2500);
            double el = (double)(GetTickCount64() - t0) / 1000.0;
            rate[n] = (double)(totalHashes - h0) / el;
            double g = (double)(gpuHashes - g0) / el;
            if (rate[n] > best) best = rate[n];
            printf("  %2d threads: %sH/s (CPU %sH/s%s%s%s)\n", n, fmtn(rate[n], a), fmtn(rate[n] - g, b),
                   useGpu ? " + GPU " : "", useGpu ? fmtn(g, c) : "", useGpu ? "H/s" : "");
        }
        if (pass == 0) {                           /* refine around the coarse winner */
            int w = 1; for (int n = 1; n <= L; n++) if (tested[n] && rate[n] > rate[w]) w = n;
            nc = 0;
            if (w - 1 >= 1) cand[nc++] = w - 1;
            if (w + 1 <= L) cand[nc++] = w + 1;
        }
    }
    int pick = L;
    for (int n = 1; n <= L; n++) if (tested[n] && rate[n] >= 0.99 * best) { pick = n; break; }
    printf("  -> using %d CPU threads (%sH/s; fewest within 1%% of the best)\n", pick, fmtn(rate[pick], a));

    /* back to idle state for live mining */
    haveJob = 0; InterlockedIncrement(&jobGen);
    Sleep(500);
    activeThreads = pick;
    offline = 0; poolDiff = 1; vrOn = 0; bestDiff = 0; totalHashes = 0; gpuHashes = 0;
    memset(&gjob, 0, sizeof gjob);
    return pick;
}

static void bench(int secs) {
    char a[32], b[32], c[32], d[32];
    printf("Benchmarking %d CPU threads (%s)%s%s for %ds, offline...\n", activeThreads, (useBitslice && useAvx2) ? "AVX2 bitsliced" : useAvx2 ? "AVX2" : "scalar",
           useGpu ? " + GPU " : "", useGpu ? gpuName : "", secs);
    synthetic_job();
    if (wantVR) printf("(GPU: version rolling, %d versions per nonce)\n", vrCount);
    start_miners();
    Sleep(2000);
    LONG64 h0 = totalHashes, g0 = gpuHashes; ULONGLONG t0 = GetTickCount64();
    Sleep(secs * 1000);
    double el = (double)(GetTickCount64() - t0) / 1000.0;
    double rate = (double)(totalHashes - h0) / el, grate = (double)(gpuHashes - g0) / el;
    printf("%sH/s total", fmtn(rate, a));
    if (activeThreads) printf(" | CPU %sH/s (%sH/s per thread)", fmtn(rate - grate, b), fmtn((rate - grate) / activeThreads, d));
    if (useGpu) printf(" | GPU %sH/s", fmtn(grate, c));
    printf("\n");
    printf("At network difficulty ~140T: expected time to a block ~%s years\n", fmtn(1.4e14 * 4294967296.0 / rate / 31557600.0, c));
}

/* ======================= main ======================= */
static void usage(void) {
    printf("\npcsolo - Bitcoin solo lottery miner (CPU, native)\n\n"
           "  pcsolo --address <BTC address> [options]\n\n"
           "  -a, --address <addr>   your BTC receive address (public address only!)\n"
           "  -w, --worker <name>    worker name shown on the pool (default: pc)\n"
           "  -p, --pool <host:port> or a preset: public-pool (default), ckpool\n"
           "  -t, --threads <n|auto> CPU threads (default auto: ~25 s offline calibration picks the fewest\n"
           "                          threads within 1%% of the best total CPU+GPU rate; 0 = GPU only)\n"
           "  -g, --gpu              also mine on the GPU (OpenCL)\n"
           "      --gpu-intensity <n> nonces per GPU launch = 2^n, 16..28 (default 22 = ~16 ms per launch; launches are\n"
           "                          double-buffered, so 21-25 all give full speed)\n"
           "      --gpu-device <i>   pick a GPU by index (default: first NVIDIA)\n"
           "      --diff <d>         share difficulty to suggest to the pool (default 0.05)\n"
           "      --scalar           don't use AVX2\n"
           "      --bitslice         experimental bitsliced kernels (AVX2: 256 nonces/vector, GPU: 32/uint)\n"
           "      --version-roll     GPU: BIP 310 version rolling, 4 header versions share one message\n"
           "                          schedule per nonce (needs pool support; falls back automatically)\n"
           "      --versions <n>     header versions per nonce for --version-roll: 1, 2, 4, 8 (default) or 16\n"
           "      --no-second-pre    don't precompute the second hash's first round (experiment)\n"
           "      --gpu-maxregs <n>  NVIDIA register cap for the version-rolling kernel (experiment; slower here)\n"
           "      --gpu-local <n>    fixed GPU workgroup size (experiment; driver default is fastest here)\n"
           "      --normal-priority  don't drop to idle priority\n"
           "      --bench [secs]     offline hashrate benchmark\n"
           "      --selftest         verify SHA-256d against the genesis block\n\n");
}

static int valid_address(const char *a) {
    size_t n = strlen(a);
    if (!strncmp(a, "bc1", 3)) {
        if (n < 14 || n > 90) return 0;
        for (size_t i = 3; i < n; i++) if (!strchr("qpzry9x8gf2tvdw0s3jn54khce6mua7l", a[i])) return 0;
        return 1;
    }
    if ((a[0] == '1' || a[0] == '3') && n >= 26 && n <= 35) {
        for (size_t i = 0; i < n; i++) if (!strchr("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz", a[i])) return 0;
        return 1;
    }
    return 0;
}

static void set_pool(const char *p) {
    if (!strcmp(p, "public-pool")) { strcpy(host, "public-pool.io"); strcpy(portstr, "21496"); return; }
    if (!strcmp(p, "ckpool")) { strcpy(host, "solo.ckpool.org"); strcpy(portstr, "3333"); return; }
    if (!strncmp(p, "stratum+tcp://", 14)) p += 14;
    const char *c = strrchr(p, ':');
    if (!c) { fprintf(stderr, "pool must be host:port\n"); exit(1); }
    snprintf(host, sizeof host, "%.*s", (int)(c - p), p);
    snprintf(portstr, sizeof portstr, "%s", c + 1);
}

int main(int argc, char **argv) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    nthreads = si.dwNumberOfProcessors / 2; if (nthreads < 1) nthreads = 1;
    InitializeCriticalSection(&bestCs); InitializeCriticalSection(&sendCs); InitializeCriticalSection(&logCs);
    __builtin_cpu_init(); useAvx2 = __builtin_cpu_supports("avx2");
    startTick = GetTickCount64();

    int doSelftest = 0, doBench = 0, normalPrio = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(1), ""))
        if (!strcmp(a, "-a") || !strcmp(a, "--address")) snprintf(address, sizeof address, "%s", NEXT());
        else if (!strcmp(a, "-w") || !strcmp(a, "--worker")) snprintf(worker, sizeof worker, "%s", NEXT());
        else if (!strcmp(a, "-p") || !strcmp(a, "--pool")) set_pool(NEXT());
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) {
            const char *v = NEXT();
            if (!strcmp(v, "auto")) autoThreads = 1; else { autoThreads = 0; nthreads = atoi(v); }
        }
        else if (!strcmp(a, "--diff")) suggestDiff = atof(NEXT());
        else if (!strcmp(a, "-g") || !strcmp(a, "--gpu")) useGpu = 1;
        else if (!strcmp(a, "--gpu-intensity")) { useGpu = 1; gpuIntensity = atoi(NEXT()); }
        else if (!strcmp(a, "--gpu-device")) { useGpu = 1; gpuDevice = atoi(NEXT()); }
        else if (!strcmp(a, "--scalar")) useAvx2 = 0;
        else if (!strcmp(a, "--bitslice")) useBitslice = 1;
        else if (!strcmp(a, "--version-roll")) wantVR = 1;
        else if (!strcmp(a, "--versions")) {
            wantVR = 1; vrCount = atoi(NEXT());
            if (vrCount != 1 && vrCount != 2 && vrCount != 4 && vrCount != 8 && vrCount != 16) { printf("--versions must be 1, 2, 4, 8 or 16\n"); return 1; }
            vrLog2 = __builtin_ctz(vrCount);
        }
        else if (!strcmp(a, "--second-pre")) secondPre = 1;
        else if (!strcmp(a, "--no-second-pre")) secondPre = 0;
        else if (!strcmp(a, "--gpu-maxregs")) gpuMaxRegs = atoi(NEXT());
        else if (!strcmp(a, "--gpu-local")) gpuLocal = (size_t)atoi(NEXT());
        else if (!strcmp(a, "--normal-priority")) normalPrio = 1;
        else if (!strcmp(a, "--selftest")) doSelftest = 1;
        else if (!strcmp(a, "--bench")) doBench = (i + 1 < argc && argv[i+1][0] != '-') ? atoi(argv[++i]) : 20;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help") || !strcmp(a, "/?")) { usage(); return 0; }
        else { printf("Unknown option: %s\n", a); usage(); return 1; }
    }
    if (autoThreads) nthreads = (int)si.dwNumberOfProcessors;   /* create all, auto-tune decides how many run */
    if (nthreads < 0) nthreads = 0;
    if (nthreads == 0 && !useGpu) nthreads = 1;
    if (nthreads > (int)si.dwNumberOfProcessors) nthreads = (int)si.dwNumberOfProcessors;
    activeThreads = nthreads;
    if (gpuIntensity < 16) gpuIntensity = 16;
    if (gpuIntensity > 28) gpuIntensity = 28;
    freshBuild = doSelftest || doBench;
    if (useGpu && gpu_init() != 0) { printf("GPU unavailable.\n"); return 1; }
    if (useGpu) printf("GPU: using %s, 2^%d nonces per launch\n", gpuName, gpuIntensity);
    if (useGpu && wantVR) printf("GPU: version-rolling kernel %s\n", vrRegInfo);

    if (doSelftest) return selftest();
    if (!normalPrio) SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
    if (useGpu) timeBeginPeriod(1); /* 1 ms Sleep granularity (default 15.6 ms) so the GPU feeder wakes on time */
    if (doBench) { if (autoThreads) autotune(); bench(doBench); return 0; }

    if (!address[0]) { printf("Missing --address <your BTC address>\n"); usage(); return 1; }
    if (!valid_address(address)) { printf("That does not look like a mainnet BTC address (bc1..., 1..., 3...).\n"); return 1; }
    for (char *c = worker; *c; c++) if (!isalnum((unsigned char)*c) && *c != '-' && *c != '_') { printf("Worker name: letters, digits, - and _ only.\n"); return 1; }
    snprintf(user, sizeof user, "%s.%s", address, worker);

    WSADATA wd; if (WSAStartup(MAKEWORD(2, 2), &wd)) { printf("WSAStartup failed\n"); return 1; }
    SetConsoleCtrlHandler(on_ctrl, TRUE);
    if (autoThreads) autotune();
    logf_("%s | pool %s:%s | %d CPU threads%s (%s)%s%s | payout %s", VERSION, host, portstr, activeThreads, autoThreads ? " [auto]" : "",
          (useBitslice && useAvx2) ? "AVX2 bitsliced" : useAvx2 ? "AVX2" : "scalar", useGpu ? " + GPU " : "", useGpu ? gpuName : "", address);
    start_miners();
    CloseHandle(CreateThread(NULL, 0, stats, NULL, 0, NULL));

    int backoff = 5;
    for (;;) {
        ULONGLONG t0 = GetTickCount64();
        session();
        if (GetTickCount64() - t0 > 120000) backoff = 5;
        logf_("Reconnecting in %d s...", backoff);
        Sleep(backoff * 1000);
        backoff = backoff * 2 > 120 ? 120 : backoff * 2;
    }
}
