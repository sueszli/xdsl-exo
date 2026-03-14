x86 AVX/SSE Intrinsics Reference
=================================

All files downloaded from public sources on 2026-03-14.

Intel intrinsics database:

  intel_intrinsics_data.xml
    Complete Intel intrinsics database in XML. Contains every SSE, AVX, AVX2,
    AVX-512, FMA intrinsic with C signatures, pseudocode, descriptions, and
    instruction mappings. This is the same data backing the interactive
    Intel Intrinsics Guide web app.
    Source: https://www.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide/data-latest.xml

  intel_intrinsics_guide.html
    Shell of the Intel Intrinsics Guide web app (JS SPA, not fully usable offline).
    Source: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

LLVM/Clang source files (from https://github.com/llvm/llvm-project, branch: main):

  IntrinsicsX86.td
    TableGen file defining all x86-specific LLVM intrinsics (SSE through AVX-512).
    Source: llvm/include/llvm/IR/IntrinsicsX86.td

  immintrin.h
    Master Clang header that includes all x86 SIMD headers below.
    Source: clang/lib/Headers/immintrin.h

  xmmintrin.h
    SSE intrinsics (128-bit float: __m128). _mm_add_ps, _mm_mul_ps, etc.
    Source: clang/lib/Headers/xmmintrin.h

  emmintrin.h
    SSE2 intrinsics (128-bit double/integer: __m128d, __m128i).
    Source: clang/lib/Headers/emmintrin.h

  smmintrin.h
    SSE4.1 intrinsics (dot product, blending, rounding, etc.).
    Source: clang/lib/Headers/smmintrin.h

  avxintrin.h
    AVX intrinsics (256-bit: __m256, __m256d). Widened float/double ops.
    Source: clang/lib/Headers/avxintrin.h

  avx2intrin.h
    AVX2 intrinsics (256-bit integer ops: __m256i).
    Source: clang/lib/Headers/avx2intrin.h

  fmaintrin.h
    FMA intrinsics (fused multiply-add: _mm256_fmadd_ps, etc.).
    Source: clang/lib/Headers/fmaintrin.h

  avx512fintrin.h
    AVX-512 Foundation intrinsics (512-bit: __m512, __m512d, __m512i).
    Source: clang/lib/Headers/avx512fintrin.h

  avx512vlintrin.h
    AVX-512 Vector Length extensions (AVX-512 ops on 128/256-bit vectors).
    Source: clang/lib/Headers/avx512vlintrin.h

  avx512bwintrin.h
    AVX-512 Byte/Word intrinsics (8/16-bit element operations).
    Source: clang/lib/Headers/avx512bwintrin.h

  avx512dqintrin.h
    AVX-512 Doubleword/Quadword intrinsics (32/64-bit enhanced ops).
    Source: clang/lib/Headers/avx512dqintrin.h

  avx512vnniintrin.h
    AVX-512 VNNI intrinsics (vector neural network instructions).
    Source: clang/lib/Headers/avx512vnniintrin.h
