/**
 * Default "unoptimized" functions
 **/
#if __APPLE__ == 1
#include <Accelerate/Accelerate.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>

float default_compute_peak(const float* src, uint32_t nframes, float current)
{
#if __APPLE__
	float tmpmax = 0.0f;
	vDSP_maxmgv(src, 1, &tmpmax, nframes);
	return fmaxf(current, tmpmax);
#else
	for (uint32_t i = 0; i < nframes; ++i)
	{
		current = fmaxf(current, fabsf(src[i]));
	}
	return current;
#endif
}

void default_apply_gain_to_buffer(float* dst, uint32_t nframes, float gain)
{
#if __APPLE__
	vDSP_vsmul(dst, 1, &gain, dst, 1, nframes);
#else
	for (uint32_t i = 0; i < nframes; ++i)
		dst[i] *= gain;
#endif
}

void default_mix_buffers_with_gain(float* dst, const float* src, uint32_t nframes, float gain)
{
#if __APPLE__
	vDSP_vsma(src, 1, &gain, dst, 1, dst, 1, nframes);
#else
	for (uint32_t i = 0; i < nframes; ++i)
		dst[i] = dst[i] + (src[i] * gain);
#endif
}

void default_mix_buffers_no_gain(float* dst, const float* src, uint32_t nframes)
{
#if __APPLE__
	vDSP_vadd(src, 1, dst, 1, dst, 1, nframes);
#else
	for (uint32_t i = 0; i < nframes; ++i)
		dst[i] += src[i];
#endif
}

void default_copy_vector(float* dst, const float* src, uint32_t nframes)
{
	memcpy(dst, src, nframes * sizeof(float));
}

void default_find_peaks(const float* buf, uint32_t nframes, float* minf, float* maxf)
{
#if __APPLE__
	float tmpmin = 0.0f;
	float tmpmax = 0.0f;
	vDSP_minv(buf, 1, &tmpmin, nframes);
	vDSP_maxv(buf, 1, &tmpmax, nframes);
	*minf = fminf(*minf, tmpmin);
	*maxf = fmaxf(*maxf, tmpmax);
#else
	uint32_t i;
	float a, b;

	a = *maxf;
	b = *minf;

	for (i = 0; i < nframes; i++)
	{
		a = fmaxf(buf[i], a);
		b = fminf(buf[i], b);
	}

	*maxf = a;
	*minf = b;
#endif
}
