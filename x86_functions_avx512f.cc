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

#define IS_ALIGNED_TO(ptr, bytes) (reinterpret_cast<uintptr_t>(ptr) % (bytes) == 0)

#ifndef __AVX512F__
#error "__AVX512F__ must be enabled for this module to work"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
 * @brief GCC builtin to hint the compiler that the expression is unlikely
 */
#if defined(__GNUC__) || defined(__clang__)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define UNLIKELY(x) (x)
#endif

#ifdef __cplusplus
#define C_FUNC extern "C"
#else
#define C_FUNC
#endif

// Number of bytes alignment for AVX-512F
#define N_ALIGNMENT sizeof(__m512)
// Number of floats processed per AVX-512F register
#define N_SIMD (N_ALIGNMENT / sizeof(float))

/**
 * @brief x86-64 AVX-512F optimized routine for compute peak procedure
 * @param src Pointer to source buffer
 * @param nframes Number of frames to process
 * @param current Current peak value
 * @return float New peak value
 */
C_FUNC float
x86_avx512f_compute_peak(const float* src, uint32_t nframes, float current)
{
	// Compute the next aligned pointer
	const float* src_aligned = (const float*) ALIGN_PTR_NEXT(src, N_ALIGNMENT);

	// Process misaligned samples before the first aligned address
	if (UNLIKELY(src_aligned > src))
	{
		// sign bit mask for absolute value
		const __m128 mask = _mm_set_ss(-0.0f);
		__m128 zmax = _mm_set_ss(current);
		size_t unaligned_count = src_aligned - src;

		// Handle small number of nframes
		size_t count = std::min<size_t>(unaligned_count, nframes);

		for (size_t i = 0; i < count; i++)
		{
			__m128 x0 = _mm_load_ss(src + i);
			x0 = _mm_andnot_ps(mask, x0); // absolute value
			zmax = _mm_max_ss(zmax, x0);
		}

		nframes -= count;
		current = _mm_cvtss_f32(zmax);
	}

	// Broadcast the current max values to all elements of the ZMM register
	__m512 zmax = _mm512_set1_ps(current);

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	#if 0
	if (simd_count >= 8)
	{
		const size_t n_loop = 8;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = simd_count / n_loop;
		for (size_t i = start; i < unrolled_count; ++i)
		{
			// Compute the pointers
			size_t offset = n_iter * i;
			const float* ptr = src_aligned + offset;

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (ptr + 2 * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3, x4, x5, x6, x7;

			// Load data from memory
			x0 = _mm512_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm512_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm512_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm512_load_ps(ptr + (3 * N_SIMD));
			x4 = _mm512_load_ps(ptr + (4 * N_SIMD));
			x5 = _mm512_load_ps(ptr + (5 * N_SIMD));
			x6 = _mm512_load_ps(ptr + (6 * N_SIMD));
			x7 = _mm512_load_ps(ptr + (7 * N_SIMD));

			// Compute absolute values
			x0 = _mm512_abs_ps(x0);
			x1 = _mm512_abs_ps(x1);
			x2 = _mm512_abs_ps(x2);
			x3 = _mm512_abs_ps(x3);
			x4 = _mm512_abs_ps(x4);
			x5 = _mm512_abs_ps(x5);
			x6 = _mm512_abs_ps(x6);
			x7 = _mm512_abs_ps(x7);

			// Compute the peaks
			x0 = _mm512_max_ps(x0, x1);
			x2 = _mm512_max_ps(x2, x3);
			x4 = _mm512_max_ps(x4, x5);
			x6 = _mm512_max_ps(x6, x7);
			x0 = _mm512_max_ps(x0, x2);
			x4 = _mm512_max_ps(x4, x6);
			x0 = _mm512_max_ps(x0, x4);

			zmax = _mm512_max_ps(zmax, x0);
		}

		start += unrolled_count * n_loop;
	}
	#endif

	if (simd_count >= 4)
	{
		const size_t n_loop = 4;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = (simd_count - start) / n_loop;
		for (size_t i = start; i < unrolled_count; ++i)
		{
			// Compute the pointers
			size_t offset = n_iter * i + (start * N_SIMD);
			const float* ptr = src_aligned + offset;

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (ptr +  n_loop * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3;

			// Load data from memory
			x0 = _mm512_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm512_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm512_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm512_load_ps(ptr + (3 * N_SIMD));

			// Compute absolute values
			x0 = _mm512_abs_ps(x0);
			x1 = _mm512_abs_ps(x1);
			x2 = _mm512_abs_ps(x2);
			x3 = _mm512_abs_ps(x3);

			// Compute the peaks
			x0 = _mm512_max_ps(x0, x1);
			x2 = _mm512_max_ps(x2, x3);
			x0 = _mm512_max_ps(x0, x2);

			zmax = _mm512_max_ps(zmax, x0);
		}

		start += unrolled_count * n_loop;
	}

	// Process remaining SIMD frames
	for (size_t i = start; i < simd_count; ++i)
	{
		size_t offset = N_SIMD * i;
		__m512 x0;

		x0 = _mm512_load_ps(src_aligned + offset);
		x0 = _mm512_abs_ps(x0);
		zmax = _mm512_max_ps(zmax, x0);
	}

	// Get the max of the ZMM registers
	current = _mm512_reduce_max_ps(zmax);

	// Process remaining samples
	if (nframes_rem > 0)
	{
		// sign bit mask for absolute value
		const __m128 mask = _mm_set_ss(-0.0f);
		__m128 zmax = _mm_set_ss(current);

		for (size_t frame = nframes_simd; frame < nframes; ++frame)
		{
			__m128 x0 = _mm_load_ss(src_aligned + frame);
			x0 = _mm_andnot_ps(mask, x0); // absolute value
			zmax = _mm_max_ss(zmax, x0);
		}

		current = _mm_cvtss_f32(zmax);
	}

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
	return current;
}

/**
 * @brief x86-64 AVX-512F optimized routine for find peak procedure
 * @param src Pointer to source buffer
 * @param nframes Number of frames to process
 * @param[in,out] minf Current minimum value, updated
 * @param[in,out] maxf Current maximum value, updated
 */
C_FUNC void
x86_avx512f_find_peaks(const float* src, uint32_t nframes, float* minf, float* maxf)
{
	// Compute the next aligned pointer
	const float* src_aligned = (const float*) ALIGN_PTR_NEXT(src, N_ALIGNMENT);
	float min_val = *minf;
	float max_val = *maxf;

	// Process misaligned samples before the first aligned address
	if (UNLIKELY(src_aligned > src))
	{
		__m128 zminb = _mm_set_ss(min_val);
		__m128 zmaxb = _mm_set_ss(max_val);

		size_t unaligned_count = src_aligned - src;
		size_t count = std::min<size_t>(unaligned_count, nframes);

		for (size_t i = 0; i < count; i++)
		{
			__m128 x = _mm_load_ss(src + i);
			zminb = _mm_min_ss(zminb, x);
			zmaxb = _mm_max_ss(zmaxb, x);
		}

		nframes -= count;
		min_val = _mm_cvtss_f32(zminb);
		max_val = _mm_cvtss_f32(zmaxb);
	}

	// Broadcast to all elements in register
	__m512 zmin = _mm512_set1_ps(min_val);
	__m512 zmax = _mm512_set1_ps(max_val);

	// Compute the number of SIMD frames
	size_t simd_count = nframes / N_SIMD;
	size_t nframes_simd = N_SIMD * simd_count;
	size_t nframes_rem = nframes - nframes_simd;
	size_t start = 0;

	if (simd_count >= 4)
	{
		const size_t n_loop = 4;
		const size_t n_iter = n_loop * N_SIMD;
		const size_t unrolled_count = (simd_count - start) / n_loop;
		for (size_t i = start; i < unrolled_count; ++i)
		{
			// Compute the pointers
			size_t offset = n_iter * i + (start * N_SIMD);
			const float* ptr = src_aligned + offset;

			// Prefetch distance in number of floats
			_mm_prefetch((const char*) (ptr + n_loop * n_iter), _MM_HINT_T0);

			__m512 x0, x1, x2, x3;
			__m512 zmin0, zmin1, zmin2;
			__m512 zmax0, zmax1, zmax2;

			// Load data from memory
			x0 = _mm512_load_ps(ptr + (0 * N_SIMD));
			x1 = _mm512_load_ps(ptr + (1 * N_SIMD));
			x2 = _mm512_load_ps(ptr + (2 * N_SIMD));
			x3 = _mm512_load_ps(ptr + (3 * N_SIMD));

			// Compute the minima
			zmin0 = _mm512_min_ps(x0, x1);
			zmin1 = _mm512_min_ps(x2, x3);
			zmin2 = _mm512_min_ps(zmin0, zmin1);
			zmin = _mm512_min_ps(zmin, zmin2);

			// Compute the maxima
			zmax0 = _mm512_max_ps(x0, x1);
			zmax1 = _mm512_max_ps(x2, x3);
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

	// Get min and max of the ZMM registers
	min_val = _mm512_reduce_min_ps(zmin);
	max_val = _mm512_reduce_max_ps(zmax);

	// Process remaining samples
	if (nframes_rem > 0)
	{
		__m128 zminb = _mm_set_ss(min_val);
		__m128 zmaxb = _mm_set_ss(max_val);

		for (size_t frame = nframes_simd; frame < nframes; ++frame)
		{
			__m128 x = _mm_load_ss(src_aligned + frame);
			zminb = _mm_min_ss(zminb, x);
			zmaxb = _mm_max_ss(zmaxb, x);
		}

		min_val = _mm_cvtss_f32(zminb);
		max_val = _mm_cvtss_f32(zmaxb);
	}

	*minf = min_val;
	*maxf = max_val;

#if defined(__GNUC__) && (__GNUC__ < 8)
	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
#endif
}

/**
 * @brief x86-64 AVX-512F optimized routine for apply gain routine
 * @param[in,out] dst Pointer to the destination buffer, which gets updated
 * @param nframes Number of frames (or samples) to process
 * @param gain Gain to apply
 */
C_FUNC void
x86_avx512f_apply_gain_to_buffer(float* dst, uint32_t nframes, float gain)
{
	// Convert to signed integer to prevent any arithmetic overflow errors
	int32_t frames = static_cast<int32_t>(nframes);

	// Load gain vector to all elements of XMM, YMM, and ZMM register
	// It's the same register, but used for SSE, AVX, and AVX512 calculation
	__m512 zgain = _mm512_set1_ps(gain);
	__m256 ygain = _mm512_castps512_ps256(zgain);
	__m128 xgain = _mm512_castps512_ps128(zgain);

	while (frames > 0)
	{
		if (IS_ALIGNED_TO(dst, sizeof(__m512)))
		{
			break;
		}

		if (frames >= 8 && IS_ALIGNED_TO(dst, sizeof(__m256)))
		{
			__m256 x = _mm256_load_ps(dst);
			__m256 y = _mm256_mul_ps(ygain, x);
			_mm256_store_ps(dst, y);
			dst += 8;
			frames -= 8;
			continue;
		}

		if (frames >= 4 && IS_ALIGNED_TO(dst, sizeof(__m128)))
		{
			__m128 x = _mm_load_ps(dst);
			__m128 y = _mm_mul_ps(xgain, x);
			_mm_store_ps(dst, y);
			dst += 4;
			frames -= 4;
			continue;
		}

		// Pointers are aligned to float boundaries (4 bytes)
		__m128 x = _mm_load_ss(dst);
		__m128 y = _mm_mul_ss(xgain, x);
		_mm_store_ss(dst, y);
		++dst;
		--frames;
	}

	// Process the remaining samples 128 at a time
	while (frames >= 128)
	{
#if defined(COMPILER_MSVC) || defined(COMPILER_MINGW)
		_mm_prefetch(reinterpret_cast<void const*>(dst + 128), _mm_hint(0));
#else
		__builtin_prefetch(reinterpret_cast<void const*>(dst + 128), 0, 0);
#endif
		__m512 x0 = _mm512_load_ps(dst + 0);
		__m512 x1 = _mm512_load_ps(dst + 16);
		__m512 x2 = _mm512_load_ps(dst + 32);
		__m512 x3 = _mm512_load_ps(dst + 48);
		__m512 x4 = _mm512_load_ps(dst + 64);
		__m512 x5 = _mm512_load_ps(dst + 80);
		__m512 x6 = _mm512_load_ps(dst + 96);
		__m512 x7 = _mm512_load_ps(dst + 112);

		__m512 y0 = _mm512_mul_ps(zgain, x0);
		__m512 y1 = _mm512_mul_ps(zgain, x1);
		__m512 y2 = _mm512_mul_ps(zgain, x2);
		__m512 y3 = _mm512_mul_ps(zgain, x3);
		__m512 y4 = _mm512_mul_ps(zgain, x4);
		__m512 y5 = _mm512_mul_ps(zgain, x5);
		__m512 y6 = _mm512_mul_ps(zgain, x6);
		__m512 y7 = _mm512_mul_ps(zgain, x7);

		_mm512_store_ps(dst + 0, y0);
		_mm512_store_ps(dst + 16, y1);
		_mm512_store_ps(dst + 32, y2);
		_mm512_store_ps(dst + 48, y3);
		_mm512_store_ps(dst + 64, y4);
		_mm512_store_ps(dst + 80, y5);
		_mm512_store_ps(dst + 96, y6);
		_mm512_store_ps(dst + 112, y7);

		dst += 128;
		frames -= 128;
	}

	// Process the remaining samples 16 at a time
	while (frames >= 16)
	{
		__m512 x = _mm512_load_ps(dst);
		__m512 y = _mm512_mul_ps(zgain, x);
		_mm512_store_ps(dst, y);

		dst += 16;
		frames -= 16;
	}

	// Process remaining samples x8
	while (frames >= 8)
	{
		__m256 x = _mm256_load_ps(dst);
		__m256 y = _mm256_mul_ps(ygain, x);
		_mm256_store_ps(dst, y);

		dst += 8;
		frames -= 8;
	}

	// Process remaining samples x4
	while (frames >= 4)
	{
		__m128 x = _mm_load_ps(dst);
		__m128 y = _mm_mul_ps(xgain, x);
		_mm_store_ps(dst, y);

		dst += 4;
		frames -= 4;
	}

	// Process remaining samples
	while (frames > 0)
	{
		__m128 x = _mm_load_ss(dst);
		__m128 y = _mm_mul_ss(xgain, x);
		_mm_store_ss(dst, y);
		++dst;
		--frames;
	}

	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.
	//
	_mm256_zeroupper(); // zeros the upper portion of YMM register
}

/**
 * @brief x86-64 AVX-512F optimized routine for mixing buffer with gain.
 * @param[in,out] dst Pointer to destination buffer, which gets updated
 * @param[in] src Pointer to source buffer (not updated)
 * @param nframes Number of samples to process
 * @param gain Gain to apply
 */
C_FUNC void
x86_avx512f_mix_buffers_with_gain(float* dst, const float* src, uint32_t nframes, float gain)
{
	// Convert to signed integer to prevent any arithmetic overflow errors
	int32_t frames = static_cast<int32_t>(nframes);

	// Load gain vector to all elements of XMM, YMM, and ZMM register
	// It's the same register, but used for SSE, AVX, and AVX512 calculation
	__m512 zgain = _mm512_set1_ps(gain);
	__m256 ygain = _mm512_castps512_ps256(zgain);
	__m128 xgain = _mm512_castps512_ps128(zgain);

	while (frames > 0)
	{
		if (IS_ALIGNED_TO(src, sizeof(__m512)) && IS_ALIGNED_TO(dst, sizeof(__m512)))
		{
			break;
		}

		if (frames >= 8 && IS_ALIGNED_TO(src, sizeof(__m256)) && IS_ALIGNED_TO(dst, sizeof(__m256)))
		{
			__m256 x = _mm256_load_ps(src);
			__m256 y = _mm256_load_ps(dst);

			y = _mm256_fmadd_ps(ygain, x, y);
			_mm256_store_ps(dst, y);

			src += 8;
			dst += 8;
			frames -= 8;
			continue;
		}

		if (frames >= 4 && IS_ALIGNED_TO(src, sizeof(__m128)) && IS_ALIGNED_TO(dst, sizeof(__m128)))
		{
			__m128 x = _mm_load_ps(src);
			__m128 y = _mm_load_ps(dst);

			y = _mm_fmadd_ps(xgain, x, y);
			_mm_store_ps(dst, y);

			src += 4;
			dst += 4;
			frames -= 4;
			continue;
		}

		// Pointers are aligned to float boundaries (4 bytes)
		__m128 x = _mm_load_ss(src);
		__m128 y = _mm_load_ss(dst);

		y = _mm_fmadd_ss(xgain, x, y);
		_mm_store_ss(dst, y);

		++src;
		++dst;
		--frames;
	}

	// Process the remaining samples 128 at a time
	while (frames >= 128)
	{
#if defined(COMPILER_MSVC) || defined(COMPILER_MINGW)
		_mm_prefetch(reinterpret_cast<void const*>(src + 128), _mm_hint(0));
		_mm_prefetch(reinterpret_cast<void const*>(dst + 128), _mm_hint(0));
#else
		__builtin_prefetch(reinterpret_cast<void const*>(src + 128), 0, 0);
		__builtin_prefetch(reinterpret_cast<void const*>(dst + 128), 0, 0);
#endif

		__m512 x0 = _mm512_load_ps(src + 0);
		__m512 x1 = _mm512_load_ps(src + 16);
		__m512 x2 = _mm512_load_ps(src + 32);
		__m512 x3 = _mm512_load_ps(src + 48);
		__m512 x4 = _mm512_load_ps(src + 64);
		__m512 x5 = _mm512_load_ps(src + 80);
		__m512 x6 = _mm512_load_ps(src + 96);
		__m512 x7 = _mm512_load_ps(src + 112);

		__m512 y0 = _mm512_load_ps(dst + 0);
		__m512 y1 = _mm512_load_ps(dst + 16);
		__m512 y2 = _mm512_load_ps(dst + 32);
		__m512 y3 = _mm512_load_ps(dst + 48);
		__m512 y4 = _mm512_load_ps(dst + 64);
		__m512 y5 = _mm512_load_ps(dst + 80);
		__m512 y6 = _mm512_load_ps(dst + 96);
		__m512 y7 = _mm512_load_ps(dst + 112);

		y0 = _mm512_fmadd_ps(zgain, x0, y0);
		y1 = _mm512_fmadd_ps(zgain, x1, y1);
		y2 = _mm512_fmadd_ps(zgain, x2, y2);
		y3 = _mm512_fmadd_ps(zgain, x3, y3);
		y4 = _mm512_fmadd_ps(zgain, x4, y4);
		y5 = _mm512_fmadd_ps(zgain, x5, y5);
		y6 = _mm512_fmadd_ps(zgain, x6, y6);
		y7 = _mm512_fmadd_ps(zgain, x7, y7);

		_mm512_store_ps(dst + 0, y0);
		_mm512_store_ps(dst + 16, y1);
		_mm512_store_ps(dst + 32, y2);
		_mm512_store_ps(dst + 48, y3);
		_mm512_store_ps(dst + 64, y4);
		_mm512_store_ps(dst + 80, y5);
		_mm512_store_ps(dst + 96, y6);
		_mm512_store_ps(dst + 112, y7);

		src += 128;
		dst += 128;
		frames -= 128;
	}

	// Process the remaining samples 16 at a time
	while (frames >= 16)
	{
		__m512 x = _mm512_load_ps(src);
		__m512 y = _mm512_load_ps(dst);
		y = _mm512_fmadd_ps(zgain, x, y);
		_mm512_store_ps(dst, y);

		src += 16;
		dst += 16;
		frames -= 16;
	}

	// Process remaining samples x8
	while (frames >= 8)
	{
		__m256 x = _mm256_load_ps(src);
		__m256 y = _mm256_load_ps(dst);

		y = _mm256_fmadd_ps(ygain, x, y);
		_mm256_store_ps(dst, y);

		src += 8;
		dst += 8;
		frames -= 8;
	}

	// Process remaining samples x4
	while (frames >= 4)
	{
		__m128 x = _mm_load_ps(src);
		__m128 y = _mm_load_ps(dst);

		y = _mm_fmadd_ps(xgain, x, y);
		_mm_store_ps(dst, y);

		src += 4;
		dst += 4;
		frames -= 4;
	}

	// Process remaining samples
	while (frames > 0)
	{
		__m128 x = _mm_load_ss(src);
		__m128 y = _mm_load_ss(dst);

		y = _mm_fmadd_ss(xgain, x, y);
		_mm_store_ss(dst, y);

		++src;
		++dst;
		--frames;
	}

	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.
	//
	_mm256_zeroupper(); // zeros the upper portion of YMM register
}

/**
 * @brief x86-64 AVX-512F optimized routine for mixing buffer with no gain.
 * @param[in,out] dst Pointer to destination buffer, which gets updated
 * @param[in] src Pointer to source buffer (not updated)
 * @param nframes Number of samples to process
 */
C_FUNC void
x86_avx512f_mix_buffers_no_gain(float* dst, const float* src, uint32_t nframes)
{
	// Convert to signed integer to prevent any arithmetic overflow errors
	int32_t frames = static_cast<int32_t>(nframes);

	while (frames > 0)
	{
		if (IS_ALIGNED_TO(src, sizeof(__m512)) && IS_ALIGNED_TO(dst, sizeof(__m512)))
		{
			break;
		}

		if (frames >= 8 && IS_ALIGNED_TO(src, sizeof(__m256)) && IS_ALIGNED_TO(dst, sizeof(__m256)))
		{
			__m256 x = _mm256_load_ps(src);
			__m256 y = _mm256_load_ps(dst);
			y = _mm256_add_ps(x, y);
			_mm256_store_ps(dst, y);
			src += 8;
			dst += 8;
			frames -= 8;
			continue;
		}

		if (frames >= 4 && IS_ALIGNED_TO(src, sizeof(__m128)) && IS_ALIGNED_TO(dst, sizeof(__m128)))
		{
			__m128 x = _mm_load_ps(src);
			__m128 y = _mm_load_ps(dst);
			y = _mm_add_ps(x, y);
			_mm_store_ps(dst, y);
			src += 4;
			dst += 4;
			frames -= 4;
			continue;
		}

		// Pointers are aligned to float boundaries (4 bytes)
		__m128 x = _mm_load_ss(src);
		__m128 y = _mm_load_ss(dst);
		y = _mm_add_ss(x, y);
		_mm_store_ss(dst, y);
		++src;
		++dst;
		--frames;
	}

	// Process the remaining samples 128 at a time
	while (frames >= 128)
	{
#if defined(COMPILER_MSVC) || defined(COMPILER_MINGW)
		_mm_prefetch(reinterpret_cast<void const*>(src + 128), _mm_hint(0));
		_mm_prefetch(reinterpret_cast<void const*>(dst + 128), _mm_hint(0));
#else
		__builtin_prefetch(reinterpret_cast<void const*>(src + 128), 0, 0);
		__builtin_prefetch(reinterpret_cast<void const*>(dst + 128), 0, 0);
#endif

		__m512 x0 = _mm512_load_ps(src + 0);
		__m512 x1 = _mm512_load_ps(src + 16);
		__m512 x2 = _mm512_load_ps(src + 32);
		__m512 x3 = _mm512_load_ps(src + 48);
		__m512 x4 = _mm512_load_ps(src + 64);
		__m512 x5 = _mm512_load_ps(src + 80);
		__m512 x6 = _mm512_load_ps(src + 96);
		__m512 x7 = _mm512_load_ps(src + 112);

		__m512 y0 = _mm512_load_ps(dst + 0);
		__m512 y1 = _mm512_load_ps(dst + 16);
		__m512 y2 = _mm512_load_ps(dst + 32);
		__m512 y3 = _mm512_load_ps(dst + 48);
		__m512 y4 = _mm512_load_ps(dst + 64);
		__m512 y5 = _mm512_load_ps(dst + 80);
		__m512 y6 = _mm512_load_ps(dst + 96);
		__m512 y7 = _mm512_load_ps(dst + 112);

		y0 = _mm512_add_ps(x0, y0);
		y1 = _mm512_add_ps(x1, y1);
		y2 = _mm512_add_ps(x2, y2);
		y3 = _mm512_add_ps(x3, y3);
		y4 = _mm512_add_ps(x4, y4);
		y5 = _mm512_add_ps(x5, y5);
		y6 = _mm512_add_ps(x6, y6);
		y7 = _mm512_add_ps(x7, y7);

		_mm512_store_ps(dst + 0, y0);
		_mm512_store_ps(dst + 16, y1);
		_mm512_store_ps(dst + 32, y2);
		_mm512_store_ps(dst + 48, y3);
		_mm512_store_ps(dst + 64, y4);
		_mm512_store_ps(dst + 80, y5);
		_mm512_store_ps(dst + 96, y6);
		_mm512_store_ps(dst + 112, y7);

		src += 128;
		dst += 128;
		frames -= 128;
	}

	// Process the remaining samples 16 at a time
	while (frames >= 16)
	{
		__m512 x = _mm512_load_ps(src);
		__m512 y = _mm512_load_ps(dst);

		y = _mm512_add_ps(x, y);
		_mm512_store_ps(dst, y);

		src += 16;
		dst += 16;
		frames -= 16;
	}

	// Process remaining samples x8
	while (frames >= 8)
	{
		__m256 x = _mm256_load_ps(src);
		__m256 y = _mm256_load_ps(dst);

		y = _mm256_add_ps(x, y);
		_mm256_store_ps(dst, y);

		src += 8;
		dst += 8;
		frames -= 8;
	}

	// Process remaining samples x4
	while (frames >= 4)
	{
		__m128 x = _mm_load_ps(src);
		__m128 y = _mm_load_ps(dst);

		y = _mm_add_ps(x, y);
		_mm_store_ps(dst, y);

		src += 4;
		dst += 4;
		frames -= 4;
	}

	// Process remaining samples
	while (frames > 0)
	{
		__m128 x = _mm_load_ss(src);
		__m128 y = _mm_load_ss(dst);

		y = _mm_add_ss(x, y);
		_mm_store_ss(dst, y);

		++src;
		++dst;
		--frames;
	}

	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.
	//
	_mm256_zeroupper(); // zeros the upper portion of YMM register
}

/**
 * @brief Copy vector from one location to another
 * @param[out] dst Pointer to destination buffer
 * @param[in] src Pointer to source buffer
 * @param nframes Number of samples to copy
 */
C_FUNC void
x86_avx512f_copy_vector(float* dst, const float* src, uint32_t nframes)
{
	// Convert to signed integer to prevent any arithmetic overflow errors
	int32_t frames = static_cast<int32_t>(nframes);

	while (frames > 0)
	{
		if (IS_ALIGNED_TO(src, sizeof(__m512)) && IS_ALIGNED_TO(dst, sizeof(__m512)))
		{
			break;
		}

		if (frames >= 8 && IS_ALIGNED_TO(src, sizeof(__m256)) && IS_ALIGNED_TO(dst, sizeof(__m256)))
		{
			__m256 x = _mm256_load_ps(src);
			_mm256_store_ps(dst, x);
			src += 8;
			dst += 8;
			frames -= 8;
			continue;
		}

		if (frames >= 4 && IS_ALIGNED_TO(src, sizeof(__m128)) && IS_ALIGNED_TO(dst, sizeof(__m128)))
		{
			__m128 x = _mm_load_ps(src);
			_mm_store_ps(dst, x);
			src += 4;
			dst += 4;
			frames -= 4;
			continue;
		}

		// Pointers are aligned to float boundaries (4 bytes)
		__m128 x = _mm_load_ss(src);
		_mm_store_ss(dst, x);
		++src;
		++dst;
		--frames;
	}

	// Process 256 samples at a time
	while (frames >= 256)
	{
#if defined(COMPILER_MSVC) || defined(COMPILER_MINGW)
		_mm_prefetch(reinterpret_cast<void const*>(src + 256), _mm_hint(0));
		_mm_prefetch(reinterpret_cast<void const*>(dst + 256), _mm_hint(0));
#else
		__builtin_prefetch(reinterpret_cast<void const*>(src + 256), 0, 0);
		__builtin_prefetch(reinterpret_cast<void const*>(dst + 256), 0, 0);
#endif
		__m512 x0 = _mm512_load_ps(src + 0);
		__m512 x1 = _mm512_load_ps(src + 16);
		__m512 x2 = _mm512_load_ps(src + 32);
		__m512 x3 = _mm512_load_ps(src + 48);
		__m512 x4 = _mm512_load_ps(src + 64);
		__m512 x5 = _mm512_load_ps(src + 80);
		__m512 x6 = _mm512_load_ps(src + 96);
		__m512 x7 = _mm512_load_ps(src + 112);

		__m512 x8 = _mm512_load_ps(src + 128);
		__m512 x9 = _mm512_load_ps(src + 144);
		__m512 x10 = _mm512_load_ps(src + 160);
		__m512 x11 = _mm512_load_ps(src + 176);
		__m512 x12 = _mm512_load_ps(src + 192);
		__m512 x13 = _mm512_load_ps(src + 208);
		__m512 x14 = _mm512_load_ps(src + 224);
		__m512 x15 = _mm512_load_ps(src + 240);

		_mm512_store_ps(dst + 0, x0);
		_mm512_store_ps(dst + 16, x1);
		_mm512_store_ps(dst + 32, x2);
		_mm512_store_ps(dst + 48, x3);
		_mm512_store_ps(dst + 64, x4);
		_mm512_store_ps(dst + 80, x5);
		_mm512_store_ps(dst + 96, x6);
		_mm512_store_ps(dst + 112, x7);

		_mm512_store_ps(dst + 128, x8);
		_mm512_store_ps(dst + 144, x9);
		_mm512_store_ps(dst + 160, x10);
		_mm512_store_ps(dst + 176, x11);
		_mm512_store_ps(dst + 192, x12);
		_mm512_store_ps(dst + 208, x13);
		_mm512_store_ps(dst + 224, x14);
		_mm512_store_ps(dst + 240, x15);

		src += 256;
		dst += 256;
		frames -= 256;
	}

	// Process remaining samples 64 at a time
	while (frames >= 64)
	{
#if defined(COMPILER_MSVC) || defined(COMPILER_MINGW)
		_mm_prefetch(reinterpret_cast<void const*>(src + 64), _mm_hint(0));
		_mm_prefetch(reinterpret_cast<void const*>(dst + 64), _mm_hint(0));
#else
		__builtin_prefetch(reinterpret_cast<void const*>(src + 64), 0, 0);
		__builtin_prefetch(reinterpret_cast<void const*>(dst + 64), 0, 0);
#endif

		__m512 x0 = _mm512_load_ps(src + 0);
		__m512 x1 = _mm512_load_ps(src + 16);
		__m512 x2 = _mm512_load_ps(src + 32);
		__m512 x3 = _mm512_load_ps(src + 48);

		_mm512_store_ps(dst + 0, x0);
		_mm512_store_ps(dst + 16, x1);
		_mm512_store_ps(dst + 32, x2);
		_mm512_store_ps(dst + 48, x3);

		src += 64;
		dst += 64;
		frames -= 64;
	}

	// Process remaining samples 16 at a time
	while (frames >= 16)
	{
		__m512 x = _mm512_load_ps(src);
		_mm512_store_ps(dst, x);

		src += 16;
		dst += 16;
		frames -= 16;
	}

	// Process remaining samples x8
	while (frames >= 8)
	{
		__m256 x = _mm256_load_ps(src);
		_mm256_store_ps(dst, x);

		src += 8;
		dst += 8;
		frames -= 8;
	}

	// Process remaining samples x4
	while (frames >= 4)
	{
		__m128 x = _mm_load_ps(src);
		_mm_store_ps(dst, x);

		src += 4;
		dst += 4;
		frames -= 4;
	}

	// Process remaining samples
	while (frames > 0)
	{
		__m128 x = _mm_load_ss(src);
		_mm_store_ss(dst, x);

		++src;
		++dst;
		--frames;
	}

	// There's a penalty going from AVX mode to SSE mode. This can
	// be avoided by ensuring the CPU that rest of the routine is no
	// longer interested in the upper portion of the YMM register.

	_mm256_zeroupper(); // zeros the upper portion of YMM register
}

#endif // FPU_AVX512F_SUPPORT
