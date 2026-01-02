
/**
 * A stubbed out version of Ardour's mix.h module
 */

/**
 * Default "unoptimized" functions
 **/

#pragma once

#include <cstddef>
#include <cstdint>

float
default_compute_peak(const float* src, uint32_t nframes, float current);
void
default_apply_gain_to_buffer(float* dst, uint32_t nframes, float gain);
void
default_mix_buffers_with_gain(float* dst, const float* src, uint32_t nframes, float gain);
void
default_mix_buffers_no_gain(float* dst, const float* src, uint32_t nframes);
void
default_copy_vector(float* dst, const float* src, uint32_t nframes);
void
default_find_peaks(const float* buf, uint32_t nsamples, float* minf, float* maxf);

/**
 * Optimized functions
 **/
#define LIBARDOUR_API __attribute__((visibility("default")))

extern "C"
{
	LIBARDOUR_API float
	arm_neon_compute_peak(float const* buf, uint32_t nsamples, float current);
	LIBARDOUR_API void
	arm_neon_apply_gain_to_buffer(float* buf, uint32_t nframes, float gain);
	LIBARDOUR_API void
	arm_neon_copy_vector(float* dst, float const* src, uint32_t nframes);
	LIBARDOUR_API void
	arm_neon_find_peaks(float const* src, uint32_t nframes, float* minf, float* maxf);
	LIBARDOUR_API void
	arm_neon_mix_buffers_no_gain(float* dst, float const* src, uint32_t nframes);
	LIBARDOUR_API void
	arm_neon_mix_buffers_with_gain(float* dst, float const* src, uint32_t nframes, float gain);
}

extern "C"
{
	LIBARDOUR_API float
	x86_avx512f_compute_peak(float const* buf, uint32_t nsamples, float current);
	LIBARDOUR_API void
	x86_avx512f_apply_gain_to_buffer(float* buf, uint32_t nframes, float gain);
	LIBARDOUR_API void
	x86_avx512f_mix_buffers_with_gain(float* dst, float const* src, uint32_t nframes, float gain);
	LIBARDOUR_API void
	x86_avx512f_mix_buffers_no_gain(float* dst, float const* src, uint32_t nframes);
	LIBARDOUR_API void
	x86_avx512f_copy_vector(float* dst, float const* src, uint32_t nframes);
	LIBARDOUR_API void
	x86_avx512f_find_peaks(float const* buf, uint32_t nsamples, float* min, float* max);
}

extern "C"
{
	LIBARDOUR_API float
	x86_avx_compute_peak(float const* buf, uint32_t nsamples, float current);
}

extern "C"
{
	LIBARDOUR_API float
	x86_sse_compute_peak(float const* buf, uint32_t nsamples, float current);
}
