/*
 * Copyright (C) 2026 Ayan Shafqat <ayan.x.shafqat@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
#ifdef FPU_AVX_SUPPORT

#include "ardour/mix.h"

#include <immintrin.h>

#ifndef __AVX__
#error "__AVX__ must be enabled for this module to work"
#endif

#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 * @def ALIGN_PTR_NEXT
 *
 * @brief Align a pointer to the next power-of-two boundary.
 *
 * Rounds the pointer **upwards** to the next address aligned to `align`,
 * where `align` *must* be a power of two.
 *
 * @param ptr   Pointer to be aligned.
 * @param align Alignment in bytes (must be power of two).
 *
 * @return Aligned pointer.
 */
#define ALIGN_PTR_NEXT(ptr, align) ((void*) ((((uintptr_t) (ptr)) + ((align) - 1)) & ~((uintptr_t) ((align) - 1))))

/**
 * @def UNLIKELY
 * @brief Branch prediction hint for cold/rare conditions.
 *
 * Wrap conditions that are expected to be false most of the time, e.g. handling
 * misaligned heads or short tails.
 */
#if defined(__GNUC__) || defined(__clang__)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define UNLIKELY(x) (x)
#endif

/**
 * @def C_FUNC
 * @brief C linkage for exported SIMD kernels.
 *
 * Ensures stable symbol names when compiled as C++.
 */
#ifdef __cplusplus
#define C_FUNC extern "C"
#else
#define C_FUNC
#endif

/**
 * @def N_ALIGNMENT
 * @brief Alignment (in bytes) required for aligned AVX loads/stores used in this TU.
 *
 * Defined as @c sizeof(__m256), i.e. 32 bytes.
 */
#define N_ALIGNMENT sizeof(__m256)

/**
 * @def N_SIMD
 * @brief Number of single-precision floats processed per AVX vector register.
 *
 * For AVX @c __m256 this is 8 floats.
 */
#define N_SIMD (N_ALIGNMENT / sizeof(float))

/**
 * @brief Reduce maximum
 *
 * @param v
 *
 * @return float
 */
static inline float
x86_avx_reduce_max(__m256 v)
{
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 m = _mm_max_ps(lo, hi);
	m = _mm_max_ps(m, _mm_movehl_ps(m, m));
	m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
	return _mm_cvtss_f32(m);
}

/**
 * @brief Compute the absolute peak (max |x|) of a buffer using AVX-512F.
 *
 * This kernel scans @p src and returns the maximum absolute value encountered,
 * compared against an initial running peak @p current.
 *
 * @param[in] src     Pointer to the input buffer (read-only).
 * @param[in] nframes Number of samples to process.
 * @param[in] current Initial peak value (e.g. previous max); result is at least this value.
 *
 * @return The maximum absolute sample value over the processed region.
 *
 * @note The implementation aligns @p src upward to @c N_ALIGNMENT and uses a masked
 *       vector load to handle the initial misaligned head and the final tail.
 *
 * @warning If @p nframes is 0, the function returns @p current.
 */
C_FUNC float
x86_avx_compute_peak(const float* src, uint32_t nframes, float current)
{
	// Compute the next aligned pointer
	const float* src_aligned = (const float*) ALIGN_PTR_NEXT(src, N_ALIGNMENT);

	// Broadcast the current max values to all elements of the YMM register
	const __m256 fp_sign = _mm256_set1_ps(-0.F);
	__m256 ymax = _mm256_set1_ps(current);

	// Unaligned samples to process, handle small number of nframes
	size_t unaligned_count = std::min<size_t>(src_aligned - src, nframes);
	nframes -= (uint32_t) unaligned_count;

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	// Process misaligned samples before the first aligned address
	if (UNLIKELY(src_aligned > src))
	{
		__m256 x = _mm256_loadu_ps(src);
		x = _mm256_andnot_ps(fp_sign, x);
		ymax = _mm256_max_ps(ymax, x);

		if (nframes == 0)
			goto reduce_max_ret;
	}

	// Process unrolled
	if (simd_count >= 4)
	{
		constexpr size_t n_loop = 4;
		constexpr size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;

		for (size_t i = 0; i < unrolled_count; ++i)
		{
			// Compute the pointers
			size_t offset = n_iter * i;
			const float* ptr = src_aligned + offset;

			// Prefetch the next further data
			_mm_prefetch((const char*) (ptr + 4 * n_iter), _MM_HINT_T0);

			__m256 x0, x1, x2, x3;
			__m256 max0, max1, max2;

			// Load data from memory
			x0 = _mm256_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm256_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm256_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm256_load_ps(ptr + (3 * N_SIMD));

			// Compute absolute values
			x0 = _mm256_andnot_ps(fp_sign, x0);
			x1 = _mm256_andnot_ps(fp_sign, x1);
			x2 = _mm256_andnot_ps(fp_sign, x2);
			x3 = _mm256_andnot_ps(fp_sign, x3);

			// Compute the peaks
			max0 = _mm256_max_ps(x0, x1);
			max1 = _mm256_max_ps(x2, x3);
			max2 = _mm256_max_ps(max0, max1);

			ymax = _mm256_max_ps(ymax, max2);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames 8 at a time
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m256 x;

		x = _mm256_load_ps(src_aligned + offset);
		x = _mm256_andnot_ps(fp_sign, x);
		ymax = _mm256_max_ps(ymax, x);
	}

	// Process remaining samples with scalar
	if (UNLIKELY(nframes_rem > 0))
	{
		for (size_t i = nframes_simd; i < nframes; ++i)
		{
			const float* ptr = src_aligned + nframes_simd + i;
			__m256 x;
			x = _mm256_castps128_ps256(_mm_load_ss(ptr));
			x = _mm256_andnot_ps(fp_sign, x);
			ymax = _mm256_max_ps(ymax, x);
		}
	}

reduce_max_ret:
	current = x86_avx_reduce_max(ymax);

#if defined(__GNUC__) && (__GNUC__ < 8)
	    _mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return current;
}

#endif // FPU_AVX_SUPPORT
