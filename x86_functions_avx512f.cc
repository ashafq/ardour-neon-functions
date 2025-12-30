/*
 * Copyright (C) 2023 Ayan Shafqat <ayan.x.shafqat@gmail.com>
 * Copyright (C) 2024 Robin Gareus <robin@gareus.org>
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
#ifdef FPU_AVX512F_SUPPORT

#include "ardour/mix.h"

#include <immintrin.h>
#include <xmmintrin.h>

#ifndef __AVX512F__
#error "__AVX512F__ must be enabled for this module to work"
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
 * @brief Alignment (in bytes) required for aligned AVX-512 loads/stores used in this TU.
 *
 * Defined as @c sizeof(__m512), i.e. 64 bytes.
 */
#define N_ALIGNMENT sizeof(__m512)

/**
 * @def N_SIMD
 * @brief Number of single-precision floats processed per AVX-512 vector register.
 *
 * For AVX-512F @c __m512 this is 16 floats.
 */
#define N_SIMD (N_ALIGNMENT / sizeof(float))

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
x86_avx512f_compute_peak(const float* src, uint32_t nframes, float current)
{
	// Compute the next aligned pointer
	const float* src_aligned = (const float*) ALIGN_PTR_NEXT(src, N_ALIGNMENT);

	// Broadcast the current max values to all elements of the ZMM register
	static const __m512 mask = _mm512_set1_ps(-0.0F);
	__m512 zmax = _mm512_set1_ps(current);

	// Process misaligned samples before the first aligned address
	if (UNLIKELY(src_aligned > src))
	{
		// Unaligned samples to process
		size_t unaligned_count = src_aligned - src;

		// Handle small number of nframes
		size_t count = std::min<size_t>(unaligned_count, nframes);

		__mmask16 load_mask = (1 << count) - 1;
		__m512 x0 = _mm512_maskz_loadu_ps(load_mask, src);
		x0 = _mm512_andnot_ps(mask, x0);
		zmax = _mm512_max_ps(zmax, x0);
		nframes -= (uint32_t) count;
	}

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	if (simd_count >= 4)
	{
		const size_t n_loop = 4;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;

		for (size_t i = 0; i < unrolled_count; ++i)
		{
			// Compute the pointers
			size_t offset = n_iter * i;
			const float* ptr = src_aligned + offset;

			// Prefetch the next further data
			_mm_prefetch((const char*) (ptr + 3 * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3;
			__m512 max0, max1, max2;

			// Load data from memory
			x0 = _mm512_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm512_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm512_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm512_load_ps(ptr + (3 * N_SIMD));

			// Compute absolute values
			x0 = _mm512_andnot_ps(mask, x0);
			x1 = _mm512_andnot_ps(mask, x1);
			x2 = _mm512_andnot_ps(mask, x2);
			x3 = _mm512_andnot_ps(mask, x3);

			// Compute the peaks
			max0 = _mm512_max_ps(x0, x1);
			max1 = _mm512_max_ps(x2, x3);
			max2 = _mm512_max_ps(max0, max1);

			zmax = _mm512_max_ps(zmax, max2);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames 16 at a time
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m512 x0;

		x0 = _mm512_load_ps(src_aligned + offset);
		x0 = _mm512_andnot_ps(mask, x0);
		zmax = _mm512_max_ps(zmax, x0);
	}

	// Process remaining samples, still using SIMD
	if (nframes_rem > 0)
	{
		// Create a mask for loading remaining samples
		__mmask16 load_mask = (1 << nframes_rem) - 1;
		__m512 x0 = _mm512_maskz_load_ps(load_mask, src_aligned + nframes_simd);
		x0 = _mm512_andnot_ps(mask, x0);
		zmax = _mm512_max_ps(zmax, x0);
	}

	// Get the max of the ZMM registers
	current = _mm512_reduce_max_ps(zmax);

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return current;
}

/**
 * @brief Find minimum and maximum sample values in a buffer using AVX-512F.
 *
 * Updates @p minf and @p maxf with the minimum and maximum values found in @p src.
 * The input values of @p *minf and @p *maxf are treated as initial extrema.
 *
 * @param[in] src       Pointer to the input buffer (read-only).
 * @param[in] nframes   Number of samples to process.
 * @param[in,out] minf  Pointer to running minimum; updated on return.
 * @param[in,out] maxf  Pointer to running maximum; updated on return.
 *
 * @note Uses aligned loads after aligning @p src upward to @c N_ALIGNMENT and uses masked
 *       loads for the misaligned head and tail.
 *
 * @warning @p minf and @p maxf must be valid pointers.
 */
C_FUNC void
x86_avx512f_find_peaks(const float* src, uint32_t nframes, float* minf, float* maxf)
{
	// Compute the next aligned pointer
	const float* src_aligned = (const float*) ALIGN_PTR_NEXT(src, N_ALIGNMENT);

	// Broadcast to all elements in register
	__m512 zmin = _mm512_set1_ps(*minf);
	__m512 zmax = _mm512_set1_ps(*maxf);

	// Process misaligned samples before the first aligned address
	if (UNLIKELY(src_aligned > src))
	{
		// Unaligned samples to process
		size_t unaligned_count = src_aligned - src;

		// Handle small number of nframes
		size_t count = std::min<size_t>(unaligned_count, nframes);

		// Mask load for initial unaligned samples
		__mmask16 load_mask = (1 << count) - 1;
		__m512 x0 = _mm512_maskz_loadu_ps(load_mask, src);
		zmax = _mm512_max_ps(zmax, x0);
		zmin = _mm512_min_ps(zmin, x0);
		nframes -= (uint32_t) count;
	}

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	if (simd_count >= 8)
	{
		const size_t n_loop = 8;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;

		for (size_t i = start; i < unrolled_count; ++i)
		{
			// Compute the pointer
			const float* ptr = src_aligned + n_iter * i;
			;

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (ptr + 4 * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3, x4, x5, x6, x7;
			__m512 zmin0, zmin1, zmin2, zmin3;
			__m512 zmax0, zmax1, zmax2, zmax3;

			// Load data from memory
			x0 = _mm512_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm512_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm512_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm512_load_ps(ptr + (3 * N_SIMD));
			x4 = _mm512_load_ps(ptr + (4 * N_SIMD));
			x5 = _mm512_load_ps(ptr + (5 * N_SIMD));
			x6 = _mm512_load_ps(ptr + (6 * N_SIMD));
			x7 = _mm512_load_ps(ptr + (7 * N_SIMD));

			// Compute the minima
			zmin0 = _mm512_min_ps(x0, x1);
			zmin1 = _mm512_min_ps(x2, x3);
			zmin2 = _mm512_min_ps(x4, x5);
			zmin3 = _mm512_min_ps(x6, x7);
			zmin0 = _mm512_min_ps(zmin0, zmin1);
			zmin1 = _mm512_min_ps(zmin2, zmin3);
			zmin2 = _mm512_min_ps(zmin0, zmin1);
			zmin = _mm512_min_ps(zmin, zmin2);

			// Compute the maxima
			zmax0 = _mm512_max_ps(x0, x1);
			zmax1 = _mm512_max_ps(x2, x3);
			zmax2 = _mm512_max_ps(x4, x5);
			zmax3 = _mm512_max_ps(x6, x7);
			zmax0 = _mm512_max_ps(zmax0, zmax1);
			zmax1 = _mm512_max_ps(zmax2, zmax3);
			zmax2 = _mm512_max_ps(zmax0, zmax1);
			zmax = _mm512_max_ps(zmax, zmax2);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m512 x = _mm512_load_ps(src_aligned + offset);
		zmax = _mm512_max_ps(zmax, x);
		zmin = _mm512_min_ps(zmin, x);
	}

	// Process remaining samples
	if (nframes_rem > 0)
	{
		// Create a mask for loading remaining samples
		__mmask16 load_mask = (1 << nframes_rem) - 1;
		__m512 x0 = _mm512_maskz_load_ps(load_mask, src_aligned + nframes_simd);
		zmax = _mm512_max_ps(zmax, x0);
		zmin = _mm512_min_ps(zmin, x0);
	}

	*minf = _mm512_reduce_min_ps(zmin);
	*maxf = _mm512_reduce_max_ps(zmax);

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return;
}

/**
 * @brief Multiply a buffer by a scalar gain using AVX-512F (in-place).
 *
 * Performs: @code dst[i] *= gain; @endcode for @p nframes samples.
 *
 * @param[in,out] dst   Pointer to destination buffer (modified in-place).
 * @param[in] nframes   Number of samples to process.
 * @param[in] gain      Scalar gain to apply.
 *
 * @note The implementation aligns @p dst upward to @c N_ALIGNMENT for aligned loads/stores.
 *       A masked unaligned head and masked tail handle non-multiple-of-16 sizes safely.
 */
C_FUNC void
x86_avx512f_apply_gain_to_buffer(float* dst, uint32_t nframes, float gain)
{
	// Compute the next aligned pointer
	float* dst_aligned = (float*) ALIGN_PTR_NEXT(dst, N_ALIGNMENT);

	// Broadcast gain to all elements in ZMM register
	__m512 zgain = _mm512_set1_ps(gain);

	// Process misaligned samples before the first aligned address
	if (UNLIKELY(dst_aligned > dst))
	{
		// Unaligned samples to process
		size_t unaligned_count = dst_aligned - dst;

		// Handle small number of nframes
		size_t count = std::min<size_t>(unaligned_count, nframes);

		// Load first few elements with mask
		__mmask16 load_mask = (1 << count) - 1;
		__m512 x = _mm512_maskz_loadu_ps(load_mask, dst);
		__m512 y = _mm512_mul_ps(zgain, x);

		// Store the elements back into memory
		_mm512_mask_storeu_ps(dst, load_mask, y);

		nframes -= count;
	}

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	if (simd_count >= 8)
	{
		const size_t n_loop = 8;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;

		for (size_t i = 0; i < unrolled_count; ++i)
		{
			float* ptr = dst_aligned + (i * n_iter);

			__m512 x0, x1, x2, x3, x4, x5, x6, x7;

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (ptr + 4 * n_iter), _MM_HINT_T0);

			// Load data from memory
			x0 = _mm512_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm512_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm512_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm512_load_ps(ptr + (3 * N_SIMD));
			x4 = _mm512_load_ps(ptr + (4 * N_SIMD));
			x5 = _mm512_load_ps(ptr + (5 * N_SIMD));
			x6 = _mm512_load_ps(ptr + (6 * N_SIMD));
			x7 = _mm512_load_ps(ptr + (7 * N_SIMD));

			// Multiply by gain
			x0 = _mm512_mul_ps(zgain, x0);
			x1 = _mm512_mul_ps(zgain, x1);
			x2 = _mm512_mul_ps(zgain, x2);
			x3 = _mm512_mul_ps(zgain, x3);
			x4 = _mm512_mul_ps(zgain, x4);
			x5 = _mm512_mul_ps(zgain, x5);
			x6 = _mm512_mul_ps(zgain, x6);
			x7 = _mm512_mul_ps(zgain, x7);

			// Store results back to memory
			_mm512_store_ps(ptr + (0 * N_SIMD), x0);
			_mm512_store_ps(ptr + (1 * N_SIMD), x1);
			_mm512_store_ps(ptr + (2 * N_SIMD), x2);
			_mm512_store_ps(ptr + (3 * N_SIMD), x3);
			_mm512_store_ps(ptr + (4 * N_SIMD), x4);
			_mm512_store_ps(ptr + (5 * N_SIMD), x5);
			_mm512_store_ps(ptr + (6 * N_SIMD), x6);
			_mm512_store_ps(ptr + (7 * N_SIMD), x7);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames 16 at a time
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m512 x, y;
		x = _mm512_load_ps(dst_aligned + offset);
		y = _mm512_mul_ps(zgain, x);
		_mm512_store_ps(dst_aligned + offset, y);
	}

	// Process remaining samples
	if (nframes_rem > 0)
	{
		// Create a mask for loading remaining samples
		__mmask16 load_mask = (1 << nframes_rem) - 1;
		__m512 x = _mm512_maskz_load_ps(load_mask, dst_aligned + nframes_simd);
		__m512 y = _mm512_mul_ps(zgain, x);
		_mm512_mask_store_ps(dst_aligned + nframes_simd, load_mask, y);
	}

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return;
}

/**
 * @brief Mix one buffer into another with scalar gain using AVX-512F.
 *
 * Performs: @code dst[i] += gain * src[i]; @endcode for @p nframes samples.
 *
 * @param[in,out] dst Destination buffer (accumulator), updated in-place.
 * @param[in]     src Source buffer (read-only).
 * @param[in] nframes Number of samples to process.
 * @param[in] gain    Scalar gain applied to @p src before accumulation.
 *
 * @warning @p dst and @p src are declared @c __restrict and must not overlap.
 *
 * @note This kernel uses FMA (@c _mm512_fmadd_ps) which may not be bit-identical to
 *       separate multiply+add implementations due to different rounding behavior.
 */
C_FUNC void
x86_avx512f_mix_buffers_with_gain(float* __restrict dst, const float* __restrict src, uint32_t nframes, float gain)
{
	// Broadcast to all elements in vector
	__m512 zgain = _mm512_set1_ps(gain);

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	if (simd_count >= 8)
	{
		const size_t n_loop = 8;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;

		for (size_t i = 0; i < unrolled_count; ++i)
		{
			const float* p_src = src + (i * n_iter);
			float* p_dst = dst + (i * n_iter);

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (p_src + 8 * n_iter), _MM_HINT_T0);
			_mm_prefetch((const char*) (p_dst + 8 * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3, x4, x5, x6, x7;
			__m512 y0, y1, y2, y3, y4, y5, y6, y7;

			// Load data from memory
			x0 = _mm512_loadu_ps(p_src + (0 * N_SIMD));
			x1 = _mm512_loadu_ps(p_src + (1 * N_SIMD));
			x2 = _mm512_loadu_ps(p_src + (2 * N_SIMD));
			x3 = _mm512_loadu_ps(p_src + (3 * N_SIMD));
			x4 = _mm512_loadu_ps(p_src + (4 * N_SIMD));
			x5 = _mm512_loadu_ps(p_src + (5 * N_SIMD));
			x6 = _mm512_loadu_ps(p_src + (6 * N_SIMD));
			x7 = _mm512_loadu_ps(p_src + (7 * N_SIMD));

			y0 = _mm512_loadu_ps(p_dst + (0 * N_SIMD));
			y1 = _mm512_loadu_ps(p_dst + (1 * N_SIMD));
			y2 = _mm512_loadu_ps(p_dst + (2 * N_SIMD));
			y3 = _mm512_loadu_ps(p_dst + (3 * N_SIMD));
			y4 = _mm512_loadu_ps(p_dst + (4 * N_SIMD));
			y5 = _mm512_loadu_ps(p_dst + (5 * N_SIMD));
			y6 = _mm512_loadu_ps(p_dst + (6 * N_SIMD));
			y7 = _mm512_loadu_ps(p_dst + (7 * N_SIMD));

			// y = gain * x + y
			y0 = _mm512_fmadd_ps(zgain, x0, y0);
			y1 = _mm512_fmadd_ps(zgain, x1, y1);
			y2 = _mm512_fmadd_ps(zgain, x2, y2);
			y3 = _mm512_fmadd_ps(zgain, x3, y3);
			y4 = _mm512_fmadd_ps(zgain, x4, y4);
			y5 = _mm512_fmadd_ps(zgain, x5, y5);
			y6 = _mm512_fmadd_ps(zgain, x6, y6);
			y7 = _mm512_fmadd_ps(zgain, x7, y7);

			// Store results back to memory
			_mm512_storeu_ps(p_dst + (0 * N_SIMD), y0);
			_mm512_storeu_ps(p_dst + (1 * N_SIMD), y1);
			_mm512_storeu_ps(p_dst + (2 * N_SIMD), y2);
			_mm512_storeu_ps(p_dst + (3 * N_SIMD), y3);
			_mm512_storeu_ps(p_dst + (4 * N_SIMD), y4);
			_mm512_storeu_ps(p_dst + (5 * N_SIMD), y5);
			_mm512_storeu_ps(p_dst + (6 * N_SIMD), y6);
			_mm512_storeu_ps(p_dst + (7 * N_SIMD), y7);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames 16 at a time
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m512 x, y;
		x = _mm512_loadu_ps(src + offset);
		y = _mm512_loadu_ps(dst + offset);
		y = _mm512_fmadd_ps(zgain, x, y);
		_mm512_storeu_ps(dst + offset, y);
	}

	// Process remaining samples
	if (nframes_rem > 0)
	{
		// Create a mask for loading remaining samples
		__mmask16 load_mask = (1 << nframes_rem) - 1;
		__m512 x = _mm512_maskz_loadu_ps(load_mask, src + nframes_simd);
		__m512 y = _mm512_maskz_loadu_ps(load_mask, dst + nframes_simd);
		y = _mm512_fmadd_ps(zgain, x, y);
		_mm512_mask_storeu_ps(dst + nframes_simd, load_mask, y);
	}

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return;
}

/**
 * @brief Mix one buffer into another (no gain) using AVX-512F.
 *
 * Performs: @code dst[i] += src[i]; @endcode for @p nframes samples.
 *
 * @param[in,out] dst Destination buffer (accumulator), updated in-place.
 * @param[in]     src Source buffer (read-only).
 * @param[in] nframes Number of samples to process.
 *
 * @warning @p dst and @p src are declared @c __restrict and must not overlap.
 *
 * @note Uses unaligned vector loads/stores and masked tail handling for non-multiple-of-16 sizes.
 */
C_FUNC void
x86_avx512f_mix_buffers_no_gain(float* __restrict dst, const float* __restrict src, uint32_t nframes)
{
	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	if (simd_count >= 8)
	{
		const size_t n_loop = 8;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;

		for (size_t i = 0; i < unrolled_count; ++i)
		{
			const float* p_src = src + (i * n_iter);
			float* p_dst = dst + (i * n_iter);

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (p_src + 2 * n_iter), _MM_HINT_T0);
			_mm_prefetch((const char*) (p_dst + 2 * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3, x4, x5, x6, x7;
			__m512 y0, y1, y2, y3, y4, y5, y6, y7;

			// Load data from memory
			x0 = _mm512_loadu_ps(p_src + (0 * N_SIMD));
			x1 = _mm512_loadu_ps(p_src + (1 * N_SIMD));
			x2 = _mm512_loadu_ps(p_src + (2 * N_SIMD));
			x3 = _mm512_loadu_ps(p_src + (3 * N_SIMD));
			x4 = _mm512_loadu_ps(p_src + (4 * N_SIMD));
			x5 = _mm512_loadu_ps(p_src + (5 * N_SIMD));
			x6 = _mm512_loadu_ps(p_src + (6 * N_SIMD));
			x7 = _mm512_loadu_ps(p_src + (7 * N_SIMD));

			y0 = _mm512_loadu_ps(p_dst + (0 * N_SIMD));
			y1 = _mm512_loadu_ps(p_dst + (1 * N_SIMD));
			y2 = _mm512_loadu_ps(p_dst + (2 * N_SIMD));
			y3 = _mm512_loadu_ps(p_dst + (3 * N_SIMD));
			y4 = _mm512_loadu_ps(p_dst + (4 * N_SIMD));
			y5 = _mm512_loadu_ps(p_dst + (5 * N_SIMD));
			y6 = _mm512_loadu_ps(p_dst + (6 * N_SIMD));
			y7 = _mm512_loadu_ps(p_dst + (7 * N_SIMD));

			// y = x + y
			y0 = _mm512_add_ps(x0, y0);
			y1 = _mm512_add_ps(x1, y1);
			y2 = _mm512_add_ps(x2, y2);
			y3 = _mm512_add_ps(x3, y3);
			y4 = _mm512_add_ps(x4, y4);
			y5 = _mm512_add_ps(x5, y5);
			y6 = _mm512_add_ps(x6, y6);
			y7 = _mm512_add_ps(x7, y7);

			// Store results back to memory
			_mm512_storeu_ps(p_dst + (0 * N_SIMD), y0);
			_mm512_storeu_ps(p_dst + (1 * N_SIMD), y1);
			_mm512_storeu_ps(p_dst + (2 * N_SIMD), y2);
			_mm512_storeu_ps(p_dst + (3 * N_SIMD), y3);
			_mm512_storeu_ps(p_dst + (4 * N_SIMD), y4);
			_mm512_storeu_ps(p_dst + (5 * N_SIMD), y5);
			_mm512_storeu_ps(p_dst + (6 * N_SIMD), y6);
			_mm512_storeu_ps(p_dst + (7 * N_SIMD), y7);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames 16 at a time
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m512 x, y;
		x = _mm512_loadu_ps(src + offset);
		y = _mm512_loadu_ps(dst + offset);
		y = _mm512_add_ps(x, y);
		_mm512_storeu_ps(dst + offset, y);
	}

	// Process remaining samples
	if (nframes_rem > 0)
	{
		// Create a mask for loading remaining samples
		__mmask16 load_mask = (1 << nframes_rem) - 1;
		__m512 x = _mm512_maskz_loadu_ps(load_mask, src + nframes_simd);
		__m512 y = _mm512_maskz_loadu_ps(load_mask, dst + nframes_simd);
		y = _mm512_add_ps(x, y);
		_mm512_mask_storeu_ps(dst + nframes_simd, load_mask, y);
	}

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return;
}

/**
 * @brief Copy a vector of floats from @p src to @p dst.
 *
 * Currently implemented via @c memcpy. Provided for API completeness and to keep
 * platform-specific implementations consistent across SIMD backends.
 *
 * @param[out] dst     Pointer to destination buffer.
 * @param[in]  src     Pointer to source buffer.
 * @param[in]  nframes Number of float samples to copy.
 *
 * @warning @p dst and @p src are declared @c __restrict and must not overlap.
 *
 * @note @p dst and @p src must point to valid memory regions of at least
 *       @p nframes floats.
 *
 * @note This implementation assumes @c memcpy() is optimized for AVX512, which
 *       is the case for modern version of @p glibc
 */
C_FUNC void
x86_avx512f_copy_vector(float* __restrict dst, const float* __restrict src, uint32_t nframes)
{
	memcpy(dst, src, nframes * sizeof(float));
	return;
}

#endif // FPU_AVX512F_SUPPORT
