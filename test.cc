/*
 * Copyright (C) 2023 Ayan Shafqat <ayan@shafq.at>
 * Copyright (C) 2023 Paul Davix <paul@linuxaudiosystems.com>
 * Copyright (C) 2023 Robin Gareus <robin@gareus.org>
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

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void fill_rand_f32(float* dst, size_t nframes);
float sum_abs_diff_f32(const float* src_a, const float* src_b, uint32_t nframes);
float frandf(void);

/**
 * Default "unoptimized" functions
 **/

float default_compute_peak(const float* src, uint32_t nframes, float current);
void default_apply_gain_to_buffer(float* dst, uint32_t nframes, float gain);
void default_mix_buffers_with_gain(float* dst, const float* src, uint32_t nframes, float gain);
void default_mix_buffers_no_gain(float* dst, const float* src, uint32_t nframes);
void default_copy_vector(float* dst, const float* src, uint32_t nframes);
void default_find_peaks(const float* buf, uint32_t nsamples, float* minf, float* maxf);

/**
 * Optimized AVX functions
 **/
#define LIBARDOUR_API __attribute__((visibility("default")))

extern "C"
{
	LIBARDOUR_API float arm_neon_compute_peak(float const* buf, uint32_t nsamples, float current);
	LIBARDOUR_API void arm_neon_apply_gain_to_buffer(float* buf, uint32_t nframes, float gain);
	LIBARDOUR_API void arm_neon_copy_vector(float* dst, float const* src, uint32_t nframes);
	LIBARDOUR_API void arm_neon_find_peaks(float const* src, uint32_t nframes, float* minf, float* maxf);
	LIBARDOUR_API void arm_neon_mix_buffers_no_gain(float* dst, float const* src, uint32_t nframes);
	LIBARDOUR_API void arm_neon_mix_buffers_with_gain(float* dst, float const* src, uint32_t nframes, float gain);
}

#ifdef __APPLE__
#include <mach/mach_time.h>

#define MICRO_BENCH(__statement, iter)                                                                                 \
	do                                                                                                             \
	{                                                                                                              \
		uint64_t start = mach_absolute_time();                                                                 \
		for (long i = 0; i < iter; ++i)                                                                        \
		{                                                                                                      \
			do                                                                                             \
			{                                                                                              \
				__statement                                                                            \
			} while (0);                                                                                   \
		}                                                                                                      \
		uint64_t end = mach_absolute_time();                                                                   \
                                                                                                                       \
		mach_timebase_info_data_t info;                                                                        \
		mach_timebase_info(&info);                                                                             \
		double duration = (end - start) * info.numer / (info.denom * 1e9 * iter);                              \
                                                                                                                       \
		printf("%e", duration);                                                                                \
	} while (0)

#else // __APPLE__
#include <omp.h>

#define MICRO_BENCH(__statement, iter)                                                                                 \
	do                                                                                                             \
	{                                                                                                              \
		double start = omp_get_wtime();                                                                        \
		for (long i = 0; i < iter; ++i)                                                                        \
		{                                                                                                      \
			do                                                                                             \
			{                                                                                              \
				__statement                                                                            \
			} while (0);                                                                                   \
		}                                                                                                      \
		double end = omp_get_wtime();                                                                          \
		double duration = (end - start) / ((double) iter);                                                     \
		printf("%e", duration);                                                                                \
	} while (0)
#endif // __APPLE__

#define THRESHOLD (1.0e-6)

int main(int argc, char** argv)
{
	size_t ITER = 1 << 24;

	if (argc < 3)
	{
		puts("D [num] [alignment]");
		return 1;
	}

	if (argc == 4)
	{
		ITER = atol(argv[3]);
	}

	uint32_t nframes = atoi(argv[1]);
	size_t alignment = atoi(argv[2]);
	constexpr auto ALIGNMENT = size_t(32);

	if (!nframes || alignment <= 0 || alignment >= ALIGNMENT)
	{
		puts("Invalid arguments");
		return 1;
	}

	// Ensure the allocated size is a multiple of ALIGNMENT.
	size_t extra = 2 * alignment; // extra bytes for misalignment
	size_t alloc_size = nframes * sizeof(float) + extra;
	if (alloc_size % ALIGNMENT != 0)
		alloc_size += ALIGNMENT - (alloc_size % ALIGNMENT);

	float* src_ptr = (float*) aligned_alloc(ALIGNMENT, alloc_size);
	float* dst_ptr = (float*) aligned_alloc(ALIGNMENT, alloc_size);
	float* ref_ptr = (float*) aligned_alloc(ALIGNMENT, alloc_size);

	assert(src_ptr && "src is NULL");
	assert(dst_ptr && "dst is NULL");
	assert(ref_ptr && "ref is NULL");

	float* src = src_ptr + (alignment / sizeof(float));
	float* dst = dst_ptr + (alignment / sizeof(float));
	float* ref = ref_ptr + (alignment / sizeof(float));

	srand(time(NULL));

#ifdef DEBUG_EXTRA
	printf("src address: %p\n", (void*) src);
	printf("dst address: %p\n", (void*) dst);
	printf("ref address: %p\n", (void*) ref);
#endif

	fill_rand_f32(src, nframes);
	fill_rand_f32(dst, nframes);

	printf("Function,Benchmark Time (default),Benchmark Time (ARM optimized),Unit Test,Notes\n");

	/* Unit test: Compute peak */
	{
		printf("compute_peak,");

		MICRO_BENCH({ (void) default_compute_peak(src, nframes, 0.0F); }, ITER);

		printf(",");

		MICRO_BENCH({ (void) arm_neon_compute_peak(src, nframes, 0.0F); }, ITER);

		src[5] = 5.0F;
		src[6] = -5.0F;
		float peak_d = default_compute_peak(src, nframes, 0.0F);
		float peak_a = arm_neon_compute_peak(src, nframes, 0.0F);

		if (fabsf(peak_d - peak_a) < THRESHOLD)
		{
			printf(",PASS,");
		}
		else
		{
			printf(",FAIL,");
		}

		printf("\"compute_peak [def, NEON]: %e, %e\"\n", peak_d, peak_a);
	}

	/* Unit test: find_peak */
	{
		float a, b;
		printf("find_peaks,");

		MICRO_BENCH({ (void) default_find_peaks(src, nframes, &a, &b); }, ITER);

		printf(",");

		MICRO_BENCH({ (void) arm_neon_find_peaks(src, nframes, &a, &b); }, ITER);

		float amin, bmin, amax, bmax;
		amin = bmin = __builtin_inf();
		amax = bmax = 0.0F;

		default_find_peaks(src, nframes, &amin, &amax);
		arm_neon_find_peaks(src, nframes, &bmin, &bmax);

		if ((fabsf(amin - bmin) < THRESHOLD) && (fabsf(amax - bmax) < THRESHOLD))
		{
			printf(",PASS,");
		}
		else
		{
			printf(",FAIL,");
		}

		printf("\"find_peaks [def, NEON]: (%e, %e) (%e, %e)\"\n", amin, amax, bmin, bmax);
	}

	/* Unit test: apply_gain_to_buffer */
	{
		float gain = frandf();

		printf("apply_gain_to_buffer,");

		MICRO_BENCH({ default_apply_gain_to_buffer(src, nframes, gain); }, ITER);

		printf(",");

		MICRO_BENCH({ arm_neon_apply_gain_to_buffer(src, nframes, gain); }, ITER);

		fill_rand_f32(dst, nframes);
		default_copy_vector(ref, dst, nframes);

		default_apply_gain_to_buffer(ref, nframes, gain);
		arm_neon_apply_gain_to_buffer(dst, nframes, gain);

		float err = sum_abs_diff_f32(ref, dst, nframes);

		if (err < THRESHOLD)
		{
			printf(",PASS,");
		}
		else
		{
			printf(",FAIL,");
		}

		printf("\"Error = %e\"\n", err);
	}

	/* Unit test: mix_buffers_no_gain */
	{
		float gain = frandf();

		printf("mix_buffers_no_gain,");

		MICRO_BENCH({ default_mix_buffers_no_gain(dst, src, nframes); }, ITER);

		printf(",");

		MICRO_BENCH({ arm_neon_mix_buffers_no_gain(dst, src, nframes); }, ITER);

		/* Unit test setup for Y += X */
		fill_rand_f32(src, nframes);
		fill_rand_f32(dst, nframes);
		default_copy_vector(ref, dst, nframes);

		default_mix_buffers_no_gain(ref, src, nframes);
		arm_neon_mix_buffers_no_gain(dst, src, nframes);

		float err = sum_abs_diff_f32(ref, dst, nframes);

		if (err < THRESHOLD)
		{
			printf(",PASS,");
		}
		else
		{
			printf(",FAIL,");
		}

		printf("\"Error = %e\"\n", err);
	}

	/* Unit test: mix_buffers_with_gain a.k.a saxpy */
	{
		float gain = frandf();

		printf("mix_buffers_with_gain,");

		MICRO_BENCH({ default_mix_buffers_with_gain(dst, src, nframes, gain); }, ITER);

		printf(",");

		MICRO_BENCH({ arm_neon_mix_buffers_with_gain(dst, src, nframes, gain); }, ITER);

		/* Unit test setup */
		fill_rand_f32(src, nframes);
		fill_rand_f32(dst, nframes);
		default_copy_vector(ref, dst, nframes);

		default_mix_buffers_with_gain(ref, src, nframes, gain);
		arm_neon_mix_buffers_with_gain(dst, src, nframes, gain);

		float err = sum_abs_diff_f32(ref, dst, nframes);

		if (err < THRESHOLD)
		{
			printf(",PASS,");
		}
		else
		{
			printf(",FAIL,");
		}

		printf("\"Error = %e\"\n", err);
	}

	/* Unit test: copy_vector */
	{
		printf("copy_vector,");

		MICRO_BENCH({ default_copy_vector(dst, src, nframes); }, ITER);

		printf(",");

		MICRO_BENCH({ arm_neon_copy_vector(dst, src, nframes); }, ITER);

		/* Unit test setup */
		fill_rand_f32(src, nframes);
		default_copy_vector(ref, src, nframes);
		arm_neon_copy_vector(dst, src, nframes);

		float err = sum_abs_diff_f32(ref, dst, nframes);

		if (err == 0.0F)
		{
			printf(",PASS,");
		}
		else
		{
			printf(",FAIL,");
		}

		printf("\"Error = %e\"\n", err);
	}

	free(src_ptr);
	free(dst_ptr);
	free(ref_ptr);
}

float frandf(void)
{
	const float scale = 1.0F / ((float) RAND_MAX);
	return scale * ((float) (rand()));
}

void fill_rand_f32(float* dst, size_t nframes)
{
	const float scale = 2.0F / ((float) RAND_MAX);

	for (size_t i = 0; i < nframes; ++i)
	{
		float rval = rand();
		dst[i] = rval * scale - 1.0F;
	}
}

float sum_abs_diff_f32(const float* src_a, const float* src_b, uint32_t nframes)
{
	float sum = 0.0F;

	for (uint32_t i = 0; i < nframes; ++i)
	{
		float diff = fabsf(src_a[i] - src_b[i]);

#ifdef DEBUG_EXTRA
		if (diff > 0.0F)
		{
			printf("DIFF: [a: %e, b: %e] %e\n", src_a[i], src_b[i], diff);
		}
#endif

		sum += diff;
	}

	return sum;
}
