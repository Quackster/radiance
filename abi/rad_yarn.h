#ifndef RAD_YARN_H
#define RAD_YARN_H

/* Static YaRN coefficients. The architecture resolves the correction bounds using
 * the ORIGINAL training length, never the deployment/table length. Kernels share
 * the resolved coefficients so direct and table-backed RoPE cannot disagree. */
#include "rad_plugin.h"
#include <cmath>

#if defined(__HIPCC__)
#define RAD_YARN_HD __host__ __device__
#else
#define RAD_YARN_HD
#endif

struct RadYarn {
    float factor = 0.0f;   /* zero: keep the legacy arithmetic exactly */
    float low = 0.0f;
    float high = 0.0f;
    float magnitude = 1.0f;

    RAD_YARN_HD float frequency(float inv, int i) const {
        if (factor == 0.0f) return inv;
        const float width = high == low ? 0.001f : high - low;
        const float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / width));
        /* Keep the blend order used by the reference YaRN implementation. */
        const float mask = 1.0f - ramp;
        return (inv / factor) * (1.0f - mask) + inv * mask;
    }
};

inline RadYarn rad_yarn_args(const RadArgs* a) {
    return { (float)rad_args_getf_or(a, "yarn_factor", 0.0),
             (float)rad_args_getf_or(a, "yarn_low", 0.0),
             (float)rad_args_getf_or(a, "yarn_high", 0.0),
             (float)rad_args_getf_or(a, "yarn_magnitude", 1.0) };
}

inline bool rad_yarn_valid(const RadYarn& y) {
    return std::isfinite(y.factor) && (y.factor == 0.0f ||
        (y.factor >= 1.0f && std::isfinite(y.low) && std::isfinite(y.high) &&
         y.high >= y.low && std::isfinite(y.magnitude) && y.magnitude > 0.0f));
}

#undef RAD_YARN_HD
#endif
