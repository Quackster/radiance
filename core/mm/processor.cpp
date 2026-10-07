/* processor.cpp -- see processor.h. */
#include "mm/processor.h"

#include "rad_internal.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace rad {
namespace mm {

/* ================================================================== configuration */

static bool read_triple(const RadModelMeta& m, const char* key, float out[3]) {
    const char* v = rad_meta_gets(&m, key, nullptr);
    if (!v || !*v) return false;
    std::istringstream is(v);
    float t[3];
    for (int i = 0; i < 3; ++i)
        if (!(is >> t[i])) return false;
    for (int i = 0; i < 3; ++i) out[i] = t[i];
    return true;
}

static bool soft_budget_ok(int32_t n) {
    return n == 70 || n == 140 || n == 280 || n == 560 || n == 1120;
}

int VisionConfig::from_meta(const RadModelMeta& m, std::string* why) {
    if (std::strcmp(rad_meta_gets(&m, "vision_config.model_type", ""), "siglip2_vision_model") == 0) {
        family = Family::Lfm2;
        max_segment_patches = rad_meta_geti(&m, "max_num_patches", 1024);
        patch = (int32_t)rad_meta_geti(&m, "vision_config.patch_size", 16);
        temporal = 1;
        merge = (int32_t)rad_meta_geti(&m, "downsample_factor", 2);
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = rad_meta_geti(&m, "preprocessor.min_image_tokens", 64) * f2;
        image_max_pixels = rad_meta_geti(&m, "preprocessor.max_image_tokens", 256) * f2;
        min_tiles = (int32_t)rad_meta_geti(&m, "preprocessor.min_tiles", 2);
        max_tiles = (int32_t)rad_meta_geti(&m, "preprocessor.max_tiles", 10);
        tile_size = (int32_t)rad_meta_geti(&m, "preprocessor.tile_size", 512);
        auto flag = [&](const char* k, bool def) {
            const char* v = rad_meta_gets(&m, k, def ? "true" : "false");
            return std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0;
        };
        split_images = flag("preprocessor.do_image_splitting", true);
        use_thumbnail = flag("preprocessor.use_thumbnail", true);
        max_pixels_tolerance = rad_meta_getf(&m, "preprocessor.max_pixels_tolerance", 2.0);
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        read_triple(m, "preprocessor.image_mean", mean);
        read_triple(m, "preprocessor.image_std", stdv);
        return check(why);
    }
    const char* mt = rad_meta_gets(&m, "vision_config.model_type", "");
    if (mt && !std::strcmp(mt, "gemma4_vision")) {
        family   = Family::Gemma4;
        patch    = (int32_t)rad_meta_geti(&m, "vision_config.patch_size", 0);
        merge    = (int32_t)rad_meta_geti(&m, "vision_config.pooling_kernel_size", 0);
        temporal = 1;
        if (patch <= 0 || merge <= 0) {
            if (why) *why = "the container states no vision tower geometry (vision_config."
                            "patch_size, pooling_kernel_size)";
            return RAD_E_NOTFOUND;
        }
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        tok_video = (int32_t)rad_meta_geti(&m, "video_token_id", -1);
        tok_start = (int32_t)rad_meta_geti(&m, "boi_token_id", -1);
        tok_end   = (int32_t)rad_meta_geti(&m, "eoi_token_id", -1);
        if (tok_image < 0 || tok_video < 0 || tok_start < 0 || tok_end < 0) {
            if (why) *why = "the container states no media token ids (image_token_id, "
                            "video_token_id, boi_token_id, eoi_token_id)";
            return RAD_E_NOTFOUND;
        }
        /* The budgets from the processor config, the model's stated default when it is absent. */
        image_soft_tokens = (int32_t)rad_meta_geti(
            &m, "processor.image_processor.max_soft_tokens",
            rad_meta_geti(&m, "vision_soft_tokens_per_image",
                          rad_meta_geti(&m, "vision_config.default_output_length", 280)));
        video_soft_tokens = (int32_t)rad_meta_geti(&m, "processor.video_processor.max_soft_tokens",
                                                   70);
        fps        = rad_meta_getf(&m, "processor.video_processor.fps", 1.0);
        min_frames = 1;
        max_frames = (int32_t)rad_meta_geti(&m, "processor.video_processor.max_frames", 32);
        max_frame_rows = 0;
        /* The pixel bands are QwenVL's; here they only have to be consistent for check(). */
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = video_min_pixels = f2;
        image_max_pixels = (int64_t)image_soft_tokens * f2;
        video_max_pixels = (int64_t)video_soft_tokens * f2 * max_frames;
        std::string bad;
        if (check(&bad) < 0) {
            if (why) *why = "the container's processor configuration is inconsistent: " + bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    family   = Family::QwenVL;
    patch    = (int32_t)rad_meta_geti(&m, "vision_config.patch_size", 0);
    temporal = (int32_t)rad_meta_geti(&m, "vision_config.temporal_patch_size", 0);
    merge    = (int32_t)rad_meta_geti(&m, "vision_config.spatial_merge_size", 0);
    if (patch <= 0 || temporal <= 0 || merge <= 0) {
        if (why) *why = "the container states no vision tower geometry (vision_config.patch_size, "
                        "temporal_patch_size, spatial_merge_size)";
        return RAD_E_NOTFOUND;
    }
    tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
    tok_video = (int32_t)rad_meta_geti(&m, "video_token_id", -1);
    tok_start = (int32_t)rad_meta_geti(&m, "vision_start_token_id", -1);
    tok_end   = (int32_t)rad_meta_geti(&m, "vision_end_token_id", -1);
    if (tok_image < 0 || tok_video < 0 || tok_start < 0 || tok_end < 0) {
        if (why) *why = "the container states no media token ids (image_token_id, video_token_id, "
                        "vision_start_token_id, vision_end_token_id)";
        return RAD_E_NOTFOUND;
    }
    /* THE PROCESSOR-ONLY KEYS DEFAULT TO THE FAMILY'S VALUES. They come from the checkpoint's
     * preprocessor configs, which a container may not carry; the numbers below are what the
     * Qwen3.x checkpoints ship. */
    image_min_pixels = rad_meta_geti(&m, "preprocessor.size.shortest_edge",
                                     rad_meta_geti(&m, "preprocessor.min_pixels", 65536));
    image_max_pixels = rad_meta_geti(&m, "preprocessor.size.longest_edge",
                                     rad_meta_geti(&m, "preprocessor.max_pixels", 16777216));
    video_min_pixels = rad_meta_geti(&m, "video_preprocessor.size.shortest_edge", 4096);
    video_max_pixels = rad_meta_geti(&m, "video_preprocessor.size.longest_edge", 25165824);
    fps        = rad_meta_getf(&m, "video_preprocessor.fps", 2.0);
    min_frames = (int32_t)rad_meta_geti(&m, "video_preprocessor.min_frames", 4);
    max_frames = (int32_t)rad_meta_geti(&m, "video_preprocessor.max_frames", 768);
    max_frame_rows = (int32_t)rad_meta_geti(&m, "video_preprocessor.max_video_tokens", 768);
    read_triple(m, "preprocessor.image_mean", mean);
    read_triple(m, "preprocessor.image_std", stdv);
    std::string bad;
    if (check(&bad) < 0) {
        if (why) *why = "the container's preprocessor configuration is inconsistent: " + bad;
        return RAD_E_INVAL;
    }
    for (int c = 0; c < 3; ++c)
        if (!(stdv[c] > 0.0f)) {
            if (why) *why = "preprocessor.image_std must be positive";
            return RAD_E_INVAL;
        }
    return RAD_OK;
}

int VisionConfig::check(std::string* why) const {
    auto no = [why](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    if (image_min_pixels <= 0) return no("image_min_pixels must be positive");
    if (image_max_pixels < image_min_pixels)
        return no("image_max_pixels " + std::to_string(image_max_pixels) + " is below image_min_pixels " +
                  std::to_string(image_min_pixels));
    if (family == Family::Lfm2) {
        if (patch <= 0 || merge <= 0 || temporal != 1 || max_segment_patches <= 0) return no("invalid LFM2 vision geometry");
        if (tok_image < 0) return no("image_token_id is missing");
        if (min_tiles < 1 || max_tiles < min_tiles || max_tiles > 16 ||
            tile_size < factor() || tile_size % factor()) return no("invalid LFM2 tile configuration");
        if (!(max_pixels_tolerance > 0.0) || !std::isfinite(max_pixels_tolerance))
            return no("max_pixels_tolerance must be finite and positive");
        for (int c = 0; c < 3; ++c)
            if (!std::isfinite(mean[c]) || !(stdv[c] > 0.0f) || !std::isfinite(stdv[c]))
                return no("image_mean and image_std must be finite, and image_std positive");
        return RAD_OK;
    }
    if (video_min_pixels <= 0) return no("video_min_pixels must be positive");
    if (video_max_pixels < video_min_pixels)
        return no("video_max_pixels " + std::to_string(video_max_pixels) + " is below video_min_pixels " +
                  std::to_string(video_min_pixels));
    if (!(fps > 0.0)) return no("fps must be positive");
    if (min_frames < 1) return no("min_frames must be at least 1");
    if (max_frames < min_frames)
        return no("max_frames " + std::to_string(max_frames) + " is below min_frames " +
                  std::to_string(min_frames));
    if (family == Family::Gemma4) {
        if (!soft_budget_ok(image_soft_tokens))
            return no("the image soft-token budget " + std::to_string(image_soft_tokens) +
                      " is not one of 70, 140, 280, 560, 1120");
        if (!soft_budget_ok(video_soft_tokens))
            return no("the video frame soft-token budget " + std::to_string(video_soft_tokens) +
                      " is not one of 70, 140, 280, 560, 1120");
    }
    return RAD_OK;
}

/* ================================================================== sizes and sampling */

/* Python's round(), which is round-half-to-even on the double -- the reference computes these in
 * Python, and a size that lands on .5 rounds the way it does there. */
static inline int64_t pyround(double x) { return (int64_t)std::nearbyint(x); }

int smart_resize_image(int64_t h, int64_t w, int32_t factor, int64_t min_px, int64_t max_px,
                       int64_t* oh, int64_t* ow, std::string* why) {
    if (h <= 0 || w <= 0) { if (why) *why = "the picture has no size"; return RAD_E_INVAL; }
    if ((double)std::max(h, w) / (double)std::min(h, w) > 200.0) {
        if (why) *why = "the picture's aspect ratio is over 200:1";
        return RAD_E_INVAL;
    }
    const double f = factor;
    int64_t hb = pyround((double)h / f) * factor;
    int64_t wb = pyround((double)w / f) * factor;
    if (hb * wb > max_px) {
        const double beta = std::sqrt((double)(h * w) / (double)max_px);
        hb = std::max<int64_t>(factor, (int64_t)std::floor((double)h / beta / f) * factor);
        wb = std::max<int64_t>(factor, (int64_t)std::floor((double)w / beta / f) * factor);
    } else if (hb * wb < min_px) {
        const double beta = std::sqrt((double)min_px / (double)(h * w));
        hb = (int64_t)std::ceil((double)h * beta / f) * factor;
        wb = (int64_t)std::ceil((double)w * beta / f) * factor;
    }
    *oh = hb;
    *ow = wb;
    return RAD_OK;
}

int smart_resize_video(int64_t t, int64_t h, int64_t w, int32_t tfactor, int32_t factor,
                       int64_t min_px, int64_t max_px, int64_t* oh, int64_t* ow, std::string* why) {
    if (t < tfactor) { if (why) *why = "too few frames"; return RAD_E_INVAL; }
    if (h < factor || w < factor) {
        const double s = std::max((double)factor / (double)h, (double)factor / (double)w);
        h = (int64_t)((double)h * s);
        w = (int64_t)((double)w * s);
    }
    if (h <= 0 || w <= 0) { if (why) *why = "the video has no size"; return RAD_E_INVAL; }
    if ((double)std::max(h, w) / (double)std::min(h, w) > 200.0) {
        if (why) *why = "the video's aspect ratio is over 200:1";
        return RAD_E_INVAL;
    }
    const double f = factor;
    int64_t hb = pyround((double)h / f) * factor;
    int64_t wb = pyround((double)w / f) * factor;
    const int64_t tb = pyround((double)t / (double)tfactor) * tfactor;
    if (tb * hb * wb > max_px) {
        const double beta = std::sqrt((double)(t * h * w) / (double)max_px);
        hb = std::max<int64_t>(factor, (int64_t)std::floor((double)h / beta / f) * factor);
        wb = std::max<int64_t>(factor, (int64_t)std::floor((double)w / beta / f) * factor);
    } else if (tb * hb * wb < min_px) {
        const double beta = std::sqrt((double)min_px / (double)(t * h * w));
        hb = (int64_t)std::ceil((double)h * beta / f) * factor;
        wb = (int64_t)std::ceil((double)w * beta / f) * factor;
    }
    *oh = hb;
    *ow = wb;
    return RAD_OK;
}

/* THE REFERENCE'S FRAME CHOICE: as many frames as `fps` gives over the clip's duration, clamped to
 * [min_frames, max_frames] and to what the stream has, spread evenly over it -- numpy's linspace,
 * rounded half-to-even. A stream that states no rate is taken at 24, as the reference does. */
std::vector<int64_t> sample_frames(const VideoInfo& info, double fps, int32_t min_frames,
                                   int32_t max_frames) {
    const int64_t total = info.n_frames;
    if (total <= 0) return {};
    const double vfps = info.fps > 0.0 ? info.fps : 24.0;
    int64_t n = (int64_t)((double)total / vfps * fps);
    n = std::min<int64_t>(std::max<int64_t>(n, min_frames), max_frames);
    n = std::min<int64_t>(n, total);
    std::vector<int64_t> idx((size_t)n);
    if (n == 1) { idx[0] = 0; return idx; }
    const double step = (double)(total - 1) / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) idx[(size_t)i] = (int64_t)std::nearbyint((double)i * step);
    idx[(size_t)n - 1] = total - 1;
    return idx;
}

int gemma4_size(int64_t h, int64_t w, int32_t patch, int32_t pool, int64_t max_patches,
                int64_t* oh, int64_t* ow, std::string* why) {
    if (h <= 0 || w <= 0) { if (why) *why = "the picture has no size"; return RAD_E_INVAL; }
    /* get_aspect_ratio_preserving_size, in Python's doubles. */
    const double total = (double)h * (double)w;
    const double target = (double)max_patches * (double)patch * (double)patch;
    const double factor = std::sqrt(target / total);
    const int64_t side = (int64_t)patch * pool;
    int64_t th = (int64_t)std::floor(factor * (double)h / (double)side) * side;
    int64_t tw = (int64_t)std::floor(factor * (double)w / (double)side) * side;
    if (th == 0 && tw == 0) {
        if (why) *why = "the picture is too small to make one pooled patch";
        return RAD_E_INVAL;
    }
    const int64_t max_side = (max_patches / ((int64_t)pool * pool)) * side;
    if (th == 0) {
        th = side;
        tw = std::min<int64_t>((int64_t)std::floor((double)w / (double)h) * side, max_side);
    } else if (tw == 0) {
        tw = side;
        th = std::min<int64_t>((int64_t)std::floor((double)h / (double)w) * side, max_side);
    }
    if ((double)th * (double)tw > target) {
        if (why) *why = "the picture's aspect ratio does not fit the patch budget";
        return RAD_E_INVAL;
    }
    *oh = th;
    *ow = tw;
    return RAD_OK;
}

/* The reference's (EmbeddingGemma2VideoProcessor.sample_frames): the stream's duration at `fps`,
 * one frame each 1/fps seconds as int(i * native_fps / fps), then `max_frames` of those chosen by
 * numpy's linspace truncated to integers. */
std::vector<int64_t> sample_frames_gemma4(const VideoInfo& info, double fps, int32_t max_frames) {
    const int64_t total = info.n_frames;
    if (total <= 0) return {};
    std::vector<int64_t> idx;
    if (info.fps > 0.0) {
        const double step = info.fps / fps;
        const double duration = (double)total / info.fps;
        const int64_t n = std::max<int64_t>(1, (int64_t)(duration * fps));
        for (int64_t i = 0; i < n; ++i)
            idx.push_back(std::min<int64_t>(total - 1, (int64_t)((double)i * step)));
    } else {
        for (int64_t i = 0; i < total; ++i) idx.push_back(i);
    }
    if (max_frames > 0 && (int64_t)idx.size() > max_frames) {
        std::vector<int64_t> keep((size_t)max_frames);
        const double span = (double)(idx.size() - 1);
        for (int32_t i = 0; i < max_frames; ++i) {
            const double t = max_frames == 1 ? 0.0 : span * (double)i / (double)(max_frames - 1);
            keep[(size_t)i] = idx[(size_t)(int64_t)t];
        }
        idx.swap(keep);
    }
    return idx;
}

/* ================================================================== the resample
 *
 * Pillow's ImagingResample for an 8-bit RGB image, transcribed: the coefficient table in double,
 * normalised per output pixel and rounded to 22-bit fixed point; a horizontal pass over only the
 * source rows the vertical pass will read, into an 8-bit intermediate; then the vertical pass. The
 * rounding at the intermediate is part of the function -- a single-pass float resample is a
 * different (and slightly better) function, and not the one the model was trained against. */
namespace {

constexpr int kPrec = 32 - 8 - 2;

inline double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

inline uint8_t clip8(int32_t v, int precision) {
    const int32_t s = v >> precision;
    return (uint8_t)(s < 0 ? 0 : (s > 255 ? 255 : s));
}

/* One axis: `bounds` gets (first, count) per output pixel and `kk` its fixed-point weights, `ksize`
 * a row. */
int coeffs(int32_t in_size, int32_t out_size, std::vector<int32_t>* bounds,
           std::vector<int32_t>* kk, int* precision, bool tensor) {
    const double scale = (double)in_size / (double)out_size;
    const double fscale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * fscale;
    const int ksize = (int)std::ceil(support) * 2 + 1;
    std::vector<double> k((size_t)ksize), weights((size_t)out_size * ksize);
    double largest = 0.0;
    bounds->assign((size_t)out_size * 2, 0);
    kk->assign((size_t)out_size * (size_t)ksize, 0);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / fscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        int x = 0;
        for (; x < xmax; ++x) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            k[(size_t)x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
            if (ww != 0.0) k[(size_t)x] /= ww;
        for (; x < ksize; ++x) k[(size_t)x] = 0.0;
        for (x = 0; x < ksize; ++x) {
            weights[(size_t)xx * ksize + x] = k[(size_t)x];
            largest = std::max(largest, k[(size_t)x]);
        }
        (*bounds)[(size_t)xx * 2]     = xmin;
        (*bounds)[(size_t)xx * 2 + 1] = xmax;
    }
    *precision = kPrec;
    /* torchvision's native uint8 bicubic uses int16 coefficients and chooses the
     * highest precision that fits, independently per axis. PIL uses 22 bits.
     * Both round to uint8 between the two passes. */
    if (tensor) {
        for (*precision = 0; *precision < 22; ++*precision)
            if ((int)(0.5 + largest * (1 << (*precision + 1))) >= (1 << 15)) break;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        const double v = weights[i] * (1 << *precision);
        (*kk)[i] = (int32_t)(v < 0 ? -0.5 + v : 0.5 + v);
    }
    return ksize;
}

}  /* namespace */

static void resize_bicubic_impl(const Frame& in, int32_t w, int32_t h, Frame* out, bool tensor) {
    if (in.w == w && in.h == h) { *out = in; return; }
    std::vector<int32_t> bh, kh, bv, kv;
    int ph = kPrec, pv = kPrec;
    const int ksh = coeffs(in.w, w, &bh, &kh, &ph, tensor);
    const int ksv = coeffs(in.h, h, &bv, &kv, &pv, tensor);

    /* The source rows the vertical pass reads, which is all the horizontal pass has to produce. */
    const int y_first = bv[0];
    const int y_last = bv[(size_t)h * 2 - 2] + bv[(size_t)h * 2 - 1];
    const bool need_h = w != in.w;
    const bool need_v = h != in.h;

    Frame tmp;
    const Frame* src = &in;
    if (need_h) {
        const int rows = need_v ? y_last - y_first : in.h;
        const int y0 = need_v ? y_first : 0;
        tmp.w = w;
        tmp.h = rows;
        tmp.rgb.assign((size_t)w * rows * 3, 0);
        for (int yy = 0; yy < rows; ++yy) {
            const uint8_t* row = in.rgb.data() + (size_t)(yy + y0) * in.w * 3;
            uint8_t* o = tmp.rgb.data() + (size_t)yy * w * 3;
            for (int xx = 0; xx < w; ++xx) {
                const int xmin = bh[(size_t)xx * 2], xmax = bh[(size_t)xx * 2 + 1];
                const int32_t* k = kh.data() + (size_t)xx * ksh;
                int32_t s0 = 1 << (ph - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xmax; ++x) {
                    const uint8_t* p = row + (size_t)(x + xmin) * 3;
                    s0 += p[0] * k[x];
                    s1 += p[1] * k[x];
                    s2 += p[2] * k[x];
                }
                o[xx * 3 + 0] = clip8(s0, ph);
                o[xx * 3 + 1] = clip8(s1, ph);
                o[xx * 3 + 2] = clip8(s2, ph);
            }
        }
        src = &tmp;
    }
    if (!need_v) { *out = std::move(tmp); return; }
    const int shift = need_h ? y_first : 0;
    out->w = src->w;
    out->h = h;
    out->rgb.assign((size_t)src->w * h * 3, 0);
    for (int yy = 0; yy < h; ++yy) {
        const int ymin = bv[(size_t)yy * 2] - shift, ymax = bv[(size_t)yy * 2 + 1];
        const int32_t* k = kv.data() + (size_t)yy * ksv;
        uint8_t* o = out->rgb.data() + (size_t)yy * src->w * 3;
        for (int xx = 0; xx < src->w; ++xx) {
            int32_t s0 = 1 << (pv - 1), s1 = s0, s2 = s0;
            for (int y = 0; y < ymax; ++y) {
                const uint8_t* p = src->rgb.data() + ((size_t)(y + ymin) * src->w + xx) * 3;
                s0 += p[0] * k[y];
                s1 += p[1] * k[y];
                s2 += p[2] * k[y];
            }
            o[xx * 3 + 0] = clip8(s0, pv);
            o[xx * 3 + 1] = clip8(s1, pv);
            o[xx * 3 + 2] = clip8(s2, pv);
        }
    }
}

void resize_bicubic(const Frame& in, int32_t w, int32_t h, Frame* out) {
    resize_bicubic_impl(in, w, h, out, false);
}

/* ================================================================== patches */

namespace {

inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/* Two independent 64-bit lanes over the patch bytes and the grid. The prefix cache folds the key
 * into a chained 128-bit block identity; a collision would serve one picture's cache for another,
 * so both lanes are carried. */
void content_key(const Item& it, uint64_t* hi, uint64_t* lo) {
    uint64_t a = 0x6a09e667f3bcc908ull ^ mix64((uint64_t)it.grid_t << 42 ^
                                               (uint64_t)it.grid_h << 21 ^ (uint64_t)it.grid_w);
    uint64_t b = 0xbb67ae8584caa73bull ^
                 mix64(it.kind == Kind::Image ? 1u : it.kind == Kind::Video ? 2u : 3u);
    for (const auto& seg : it.segments) {
        a = mix64(a ^ (uint64_t)seg.h << 32 ^ (uint64_t)seg.w);
        b = mix64(b + (uint64_t)seg.patch0);
    }
    const uint8_t* p = reinterpret_cast<const uint8_t*>(it.pixels.data());
    const size_t n = it.pixels.size() * sizeof(uint16_t);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t w0, w1;
        std::memcpy(&w0, p + i, 8);
        std::memcpy(&w1, p + i + 8, 8);
        a = mix64(a ^ w0) + w1;
        b = mix64(b ^ w1) ^ w0;
    }
    for (; i < n; ++i) { a = mix64(a ^ p[i]); b = mix64(b + p[i]); }
    *hi = mix64(a ^ (uint64_t)n);
    *lo = mix64(b + (uint64_t)n);
}

/* THE PATCH ROWS, in merge-block order, from `frames` (temporal * grid_t of them, already resized
 * to grid_h * patch by grid_w * patch). Each row is (channel, frame, y, x), each value normalised
 * as the reference does -- (x - 255 mean) / (255 std) in f32 -- and rounded to bf16. */
void patchify(const VisionConfig& c, const std::vector<const Frame*>& frames, Item* it) {
    const int P = c.patch, T = c.temporal, M = c.merge;
    const int gh = it->grid_h, gw = it->grid_w;
    float m255[3], s255[3];
    for (int ch = 0; ch < 3; ++ch) {
        /* torch.tensor(mean) * (1 / rescale_factor), in f32: the fused rescale's own constants. */
        m255[ch] = c.mean[ch] * (float)(1.0 / (1.0 / 255.0));
        s255[ch] = c.stdv[ch] * (float)(1.0 / (1.0 / 255.0));
    }
    /* One lookup per (channel, byte): the normalisation is a function of the byte alone. */
    uint16_t lut[3][256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v)
            lut[ch][v] = rad_f32_to_bf16(((float)v - m255[ch]) / s255[ch]);

    const int64_t dim = it->patch_dim;
    it->pixels.assign((size_t)it->n_patch * (size_t)dim, 0);
    uint16_t* dst = it->pixels.data();
    for (int g = 0; g < it->grid_t; ++g)
        for (int bh = 0; bh < gh / M; ++bh)
            for (int bw = 0; bw < gw / M; ++bw)
                for (int mh = 0; mh < M; ++mh)
                    for (int mw = 0; mw < M; ++mw) {
                        const int py0 = (bh * M + mh) * P, px0 = (bw * M + mw) * P;
                        for (int ch = 0; ch < 3; ++ch)
                            for (int t = 0; t < T; ++t) {
                                const Frame& f = *frames[(size_t)(g * T + t)];
                                for (int y = 0; y < P; ++y) {
                                    const uint8_t* row = f.rgb.data() +
                                        ((size_t)(py0 + y) * f.w + px0) * 3 + ch;
                                    for (int x = 0; x < P; ++x) *dst++ = lut[ch][row[x * 3]];
                                }
                            }
                    }
}

/* Gemma 4's patch rows: merge-block order like patchify's (each pooling window's patches are
 * consecutive, which is what lets the pooler be a mean of k^2 consecutive rows; attention and the
 * position terms read each patch's own coordinates, so the order is free), each row (y, x,
 * channel), each value 2 * (u * (1/255)) - 1 in f32 -- the processor's rescale and then the
 * model's own input scaling -- rounded to bf16. One frame a segment. */
void patchify_gemma4(const VisionConfig& c, const std::vector<const Frame*>& frames, Item* it) {
    const int P = c.patch, M = c.merge;
    const int gh = it->grid_h, gw = it->grid_w;
    uint16_t lut[256];
    for (int v = 0; v < 256; ++v) {
        const float p = (float)v * (float)(1.0 / 255.0);
        lut[v] = rad_f32_to_bf16(2.0f * (p - 0.5f));
    }
    const int64_t dim = it->patch_dim;
    it->pixels.assign((size_t)it->n_patch * (size_t)dim, 0);
    uint16_t* dst = it->pixels.data();
    for (int g = 0; g < it->grid_t; ++g) {
        const Frame& f = *frames[(size_t)g];
        for (int bh = 0; bh < gh / M; ++bh)
            for (int bw = 0; bw < gw / M; ++bw)
                for (int mh = 0; mh < M; ++mh)
                    for (int mw = 0; mw < M; ++mw) {
                        const int py0 = (bh * M + mh) * P, px0 = (bw * M + mw) * P;
                        for (int y = 0; y < P; ++y) {
                            const uint8_t* row = f.rgb.data() + ((size_t)(py0 + y) * f.w + px0) * 3;
                            for (int x = 0; x < 3 * P; ++x) *dst++ = lut[row[x]];
                        }
                    }
    }
}

void finish_item(const VisionConfig& c, Item* it) {
    it->patch_dim = c.patch_dim();
    it->n_patch = (int64_t)it->grid_t * it->grid_h * it->grid_w;
    it->rows_h = it->grid_h / c.merge;
    it->rows_w = it->grid_w / c.merge;
    it->seg_rows = it->rows_h * it->rows_w;
    it->n_seg = it->grid_t;
}

}  /* namespace */

/* ================================================================== the processor */

int Processor::init(const VisionConfig& cfg,
                    std::function<std::vector<int32_t>(const std::string&)> encode_text,
                    std::function<std::vector<int32_t>(const std::string&)> encode_special) {
    if (cfg.patch <= 0 || cfg.temporal <= 0 || cfg.merge <= 0 || cfg.max_patches <= 0)
        return RAD_E_INVAL;
    cfg_ = cfg;
    if (cfg.family == Family::Gemma4) {
        /* A budget is a patch count, and an image is one segment of one pass. */
        if ((int64_t)cfg.image_soft_tokens * cfg.merge * cfg.merge > cfg.max_patches ||
            (int64_t)cfg.video_soft_tokens * cfg.merge * cfg.merge > cfg.max_patches)
            return RAD_E_INVAL;
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    /* The largest image the encoder takes in one pass, in pixels: the band's ceiling comes down to
     * it, so smart_resize itself keeps every image inside a pass. */
    const int64_t pass_px = cfg.max_patches * cfg.patch * cfg.patch;
    if (cfg_.image_max_pixels > pass_px) cfg_.image_max_pixels = pass_px;
    if (cfg_.image_min_pixels > cfg_.image_max_pixels) cfg_.image_min_pixels = cfg_.image_max_pixels;
    if (cfg_.max_video_patches <= 0)
        cfg_.max_video_patches = cfg_.video_max_pixels / ((int64_t)cfg.patch * cfg.patch);
    if (cfg_.family == Family::Lfm2) {
        if (!encode_special) return RAD_E_INVAL;
        auto special = [&](const std::string& text) {
            const auto ids = encode_special(text);
            return ids.size() == 1 ? ids[0] : -1;
        };
        cfg_.tok_start = special("<|image_start|>");
        cfg_.tok_end = special("<|image_end|>");
        tok_thumbnail_ = special("<|img_thumbnail|>");
        if (cfg_.tok_start < 0 || cfg_.tok_end < 0 || tok_thumbnail_ < 0) return RAD_E_INVAL;
        tile_tokens_.assign((size_t)cfg_.max_tiles * cfg_.max_tiles, -1);
        for (int row = 0; row < cfg_.max_tiles; ++row)
            for (int col = 0; col < cfg_.max_tiles; ++col) {
                if ((row + 1) * (col + 1) > cfg_.max_tiles) continue;
                auto id = special("<|img_row_" + std::to_string(row + 1) + "_col_" +
                                  std::to_string(col + 1) + "|>");
                if (id < 0) return RAD_E_INVAL;
                tile_tokens_[(size_t)row * cfg_.max_tiles + col] = id;
            }
    }
    encode_text_ = std::move(encode_text);
    return RAD_OK;
}

int Processor::image_from(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (c.family == Family::Lfm2) return image_lfm2(f, out, why);
    if (c.family == Family::Gemma4) return image_gemma4(f, c.image_soft_tokens, out, why);
    int64_t h = 0, w = 0;
    RAD_TRY(smart_resize_image(f.h, f.w, c.factor(), c.image_min_pixels, c.image_max_pixels,
                               &h, &w, why));
    Frame r;
    resize_bicubic(f, (int32_t)w, (int32_t)h, &r);
    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    /* An image fills the temporal depth by repetition. */
    std::vector<const Frame*> frames((size_t)c.temporal, &r);
    patchify(c, frames, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);

    it->expand.reserve((size_t)it->seg_rows + 2);
    it->expand.push_back(c.tok_start);
    it->seg_at.push_back(1);
    it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_image);
    it->expand.push_back(c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

/* LFM2-VL uses RGB values within each patch, a variable tile grid and ordinary
 * token positions. Tiles attend independently, including the optional thumbnail. */
int Processor::image_lfm2(const Frame& input, std::shared_ptr<Item>* out, std::string* why) const {
    const auto& c = cfg_;
    auto no = [&](const char* msg) { if (why) *why = msg; return RAD_E_INVAL; };
    if (input.w <= 0 || input.h <= 0 || input.rgb.size() != (size_t)input.w * input.h * 3)
        return no("the picture has no valid RGB pixels");
    Frame capped;
    const Frame* source = &input;
    const int64_t area = (int64_t)input.w * input.h;
    if (c.cap_image_pixels > 0 && area > c.cap_image_pixels) {
        const double scale = std::sqrt((double)c.cap_image_pixels / area);
        resize_bicubic(input, std::max(1, (int)(input.w * scale)),
                       std::max(1, (int)(input.h * scale)), &capped);
        source = &capped;
    }
    const Frame& f = *source;
    const int factor = c.factor();
    int64_t h = std::max<int64_t>(factor, pyround((double)f.h / factor) * factor);
    int64_t w = std::max<int64_t>(factor, pyround((double)f.w / factor) * factor);
    const int64_t rh = std::max<int64_t>(c.patch, pyround((double)f.h / factor) * factor);
    const int64_t rw = std::max<int64_t>(c.patch, pyround((double)f.w / factor) * factor);
    const bool large = (double)(rh * rw) > c.image_max_pixels * c.max_pixels_tolerance;
    if (h * w > c.image_max_pixels) {
        const double beta = std::sqrt((double)((int64_t)f.h * f.w) / c.image_max_pixels);
        h = std::max<int64_t>(factor, (int64_t)std::floor(f.h / beta / factor) * factor);
        w = std::max<int64_t>(factor, (int64_t)std::floor(f.w / beta / factor) * factor);
    } else if (h * w < c.image_min_pixels) {
        const double beta = std::sqrt((double)c.image_min_pixels / ((int64_t)f.h * f.w));
        h = (int64_t)std::ceil(f.h * beta / factor) * factor;
        w = (int64_t)std::ceil(f.w * beta / factor) * factor;
    }
    if (h / c.patch * (w / c.patch) > std::min(c.max_patches, c.max_segment_patches))
        return no("the image aspect ratio needs more patches than the encoder segment limit");
    int cols = 1, rows = 1;
    if (large && c.split_images && c.max_tiles > 1) {
        double best = INFINITY;
        const double ratio = (double)f.w / f.h;
        /* Sorted by tile count, as the reference. Equal ratios prefer the larger
         * grid only when the original image covers more than half its area. */
        for (int n = c.min_tiles; n <= c.max_tiles; ++n)
            for (int x = 1; x <= n; ++x) {
                if (n % x) continue;
                const int y = n / x;
                const double diff = std::abs(ratio - (double)x / y);
                if (diff < best || (diff == best && (int64_t)f.w * f.h > 0.5 * c.tile_size * c.tile_size * n)) {
                    best = diff; cols = x; rows = y;
                }
            }
    }
    const bool tiled = cols * rows > 1;
    if (tiled && (int64_t)(c.tile_size / c.patch) * (c.tile_size / c.patch) > std::min(c.max_patches, c.max_segment_patches))
        return no("a tile has more patches than one encoder pass holds");
    auto it = std::make_shared<Item>();
    it->linear_positions = true;
    it->patch_dim = c.patch_dim();
    /* The entire grid is known before patchification. Reserving at each tile grows
     * and copies the already packed pixels repeatedly for a large picture. */
    const int64_t tile_patches = tiled ? (int64_t)cols * rows *
        (c.tile_size / c.patch) * (c.tile_size / c.patch) : 0;
    const int64_t thumbnail_patches = (!tiled || c.use_thumbnail) ?
        (h / c.patch) * (w / c.patch) : 0;
    it->pixels.reserve((size_t)(tile_patches + thumbnail_patches) * it->patch_dim);
    it->segments.reserve((size_t)(tiled ? cols * rows : 0) + ((!tiled || c.use_thumbnail) ? 1 : 0));
    it->expand.push_back(c.tok_start);
    uint16_t lut[3][256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v)
            lut[ch][v] = rad_f32_to_bf16(((float)v - 255.0f * c.mean[ch]) / (255.0f * c.stdv[ch]));
    auto append = [&](const Frame& frame, int x0, int y0, int width, int height) {
        const int gh = height / c.patch, gw = width / c.patch;
        it->segments.push_back({gh, gw, it->n_patch});
        it->n_patch += (int64_t)gh * gw;
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)gh * gw / (c.merge * c.merge), c.tok_image);
        for (int bh = 0; bh < gh / c.merge; ++bh)
            for (int bw = 0; bw < gw / c.merge; ++bw)
                for (int mh = 0; mh < c.merge; ++mh)
                    for (int mw = 0; mw < c.merge; ++mw)
                        for (int y = 0; y < c.patch; ++y)
                            for (int x = 0; x < c.patch; ++x)
                                for (int ch = 0; ch < 3; ++ch) {
                                    const int py = y0 + (bh * c.merge + mh) * c.patch + y;
                                    const int px = x0 + (bw * c.merge + mw) * c.patch + x;
                                    it->pixels.push_back(lut[ch][frame.rgb[((size_t)py * frame.w + px) * 3 + ch]]);
                                }
    };
    if (tiled) {
        Frame big;
        resize_bicubic_impl(f, cols * c.tile_size, rows * c.tile_size, &big, true);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x) {
                it->expand.push_back(tile_tokens_[(size_t)y * c.max_tiles + x]);
                append(big, x * c.tile_size, y * c.tile_size, c.tile_size, c.tile_size);
            }
    }
    if (!tiled || c.use_thumbnail) {
        Frame small;
        resize_bicubic_impl(f, (int32_t)w, (int32_t)h, &small, true);
        if (tiled) it->expand.push_back(tok_thumbnail_);
        append(small, 0, 0, small.w, small.h);
    }
    it->expand.push_back(c.tok_end);
    it->grid_t = it->n_seg = (int32_t)it->segments.size();
    it->grid_h = it->segments[0].h;
    it->grid_w = it->segments[0].w;
    it->rows_h = it->grid_h / c.merge;
    it->rows_w = it->grid_w / c.merge;
    it->seg_rows = it->rows_h * it->rows_w;
    content_key(*it, &it->key_hi, &it->key_lo);
    *out = std::move(it);
    return RAD_OK;
}

int Processor::image_gemma4(const Frame& f, int32_t soft, std::shared_ptr<Item>* out,
                            std::string* why) const {
    const VisionConfig& c = cfg_;
    int64_t h = 0, w = 0;
    RAD_TRY(gemma4_size(f.h, f.w, c.patch, c.merge, (int64_t)soft * c.merge * c.merge, &h, &w,
                        why));
    Frame r;
    const Frame* src = &f;
    if (h != f.h || w != f.w) {
        resize_bicubic(f, (int32_t)w, (int32_t)h, &r);
        src = &r;
    }
    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image's soft-token budget is more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    patchify_gemma4(c, { src }, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);
    it->expand.reserve((size_t)it->seg_rows + 2);
    it->expand.push_back(c.tok_start);
    it->seg_at.push_back(1);
    it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_image);
    it->expand.push_back(c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

int Processor::video_gemma4(const Video& v, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (v.frames.empty()) { if (why) *why = "the video decoded to no frames"; return RAD_E_INVAL; }
    int64_t h = 0, w = 0;
    RAD_TRY(gemma4_size(v.frames[0].h, v.frames[0].w, c.patch, c.merge,
                        (int64_t)c.video_soft_tokens * c.merge * c.merge, &h, &w, why));
    auto it = std::make_shared<Item>();
    it->kind = Kind::Video;
    it->grid_t = (int32_t)v.frames.size();
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if ((int64_t)it->grid_h * it->grid_w > c.max_patches) {
        if (why) *why = "a video frame's soft-token budget is more patches than one encoder pass "
                        "holds";
        return RAD_E_INVAL;
    }
    std::vector<Frame> resized(v.frames.size());
    std::vector<const Frame*> rf;
    for (size_t i = 0; i < v.frames.size(); ++i) {
        if (v.frames[i].h == h && v.frames[i].w == w) { rf.push_back(&v.frames[i]); continue; }
        resize_bicubic(v.frames[i], (int32_t)w, (int32_t)h, &resized[i]);
        rf.push_back(&resized[i]);
    }
    patchify_gemma4(c, rf, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);
    /* One <boi> placeholders <eoi> run a frame, back to back: the reference's visual-only form
     * (add_timestamps off). */
    for (int32_t g = 0; g < it->grid_t; ++g) {
        it->expand.push_back(c.tok_start);
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_video);
        it->expand.push_back(c.tok_end);
    }
    *out = std::move(it);
    return RAD_OK;
}

int Processor::video_from(const Video& v, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (c.family == Family::Lfm2) {
        if (why) *why = "the LFM2 processor supports images only";
        return RAD_E_UNSUPPORTED;
    }
    if (c.family == Family::Gemma4) return video_gemma4(v, out, why);
    if (v.frames.empty()) { if (why) *why = "the video decoded to no frames"; return RAD_E_INVAL; }
    /* The frames and their indices, padded to whole temporal groups by repeating the last. A clip
     * shorter than one group is padded the same way, so a single-frame video is one group. */
    std::vector<const Frame*> frames;
    std::vector<int64_t> idx = v.index;
    for (const Frame& f : v.frames) frames.push_back(&f);
    while (frames.size() % (size_t)c.temporal != 0 || frames.size() < (size_t)c.temporal) {
        frames.push_back(frames.back());
        idx.push_back(idx.back());
    }
    const int64_t nf = (int64_t)v.frames.size() < c.temporal ? c.temporal
                                                              : (int64_t)v.frames.size();

    /* THE PER-FRAME CAP: at most `max_frame_rows` output rows' worth of pixels a frame, and the
     * video budget's even share, but never under the band's floor. */
    int64_t max_px = c.video_max_pixels;
    if (c.max_frame_rows > 0) {
        const int64_t f2 = (int64_t)c.factor() * c.factor();
        const int64_t cap = (int64_t)c.max_frame_rows * f2;
        const int64_t share = c.video_max_pixels / nf;
        const int64_t per = std::max<int64_t>(std::min(cap, share),
                                              (int64_t)((double)c.video_min_pixels * 1.05));
        max_px = per * nf;
    }
    int64_t h = 0, w = 0;
    RAD_TRY(smart_resize_video(nf, v.frames[0].h, v.frames[0].w, c.temporal, c.factor(),
                               c.video_min_pixels, max_px, &h, &w, why));

    auto it = std::make_shared<Item>();
    it->kind = Kind::Video;
    it->grid_t = (int32_t)(frames.size() / (size_t)c.temporal);
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if ((int64_t)it->grid_h * it->grid_w > c.max_patches) {
        if (why) *why = "a video frame resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    if (it->n_patch > c.max_video_patches) {
        if (why) *why = "the video is over the server's patch budget for one video";
        return RAD_E_INVAL;
    }

    /* Resize each distinct frame once; padding repeats share the resized copy. */
    std::vector<Frame> resized(v.frames.size());
    for (size_t i = 0; i < v.frames.size(); ++i)
        resize_bicubic(v.frames[i], (int32_t)w, (int32_t)h, &resized[i]);
    std::vector<const Frame*> rframes;
    for (const Frame* f : frames) rframes.push_back(&resized[(size_t)(f - v.frames.data())]);
    patchify(c, rframes, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);

    /* THE TIMESTAMPED EXPANSION: per temporal group, "<T seconds>" then a run of pads between
     * vision_start and vision_end, T being the mean of the group's first and last frame times.
     * It stands for the template's whole <|vision_start|><|video_pad|><|vision_end|>, as the
     * model's own processor wrote it (transformers 4.57, qwen-vl-utils). transformers 5.x swaps
     * only the pad and leaves that outer pair around the frames -- two tokens the model was not
     * trained with. */
    const double vfps = v.info.fps > 0.0 ? v.info.fps : 24.0;
    for (int32_t g = 0; g < it->grid_t; ++g) {
        const double t0 = (double)idx[(size_t)g * c.temporal] / vfps;
        const double t1 = (double)idx[(size_t)g * c.temporal + c.temporal - 1] / vfps;
        char ts[64];
        std::snprintf(ts, sizeof ts, "<%.1f seconds>", (t0 + t1) / 2.0);
        const std::vector<int32_t> tt = encode_text_ ? encode_text_(ts) : std::vector<int32_t>{};
        if (tt.empty()) {
            if (why) *why = "the tokenizer produced nothing for a video timestamp";
            return RAD_E_STATE;
        }
        it->expand.insert(it->expand.end(), tt.begin(), tt.end());
        it->expand.push_back(c.tok_start);
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_video);
        it->expand.push_back(c.tok_end);
    }
    *out = std::move(it);
    return RAD_OK;
}

int Processor::image(const uint8_t* bytes, size_t n, int32_t image_tokens,
                     std::shared_ptr<Item>* out, std::string* why) const {
    if (image_tokens != 0) {
        if (cfg_.family != Family::Gemma4) {
            if (why) *why = "this model's processor sizes an image by its pixels, not by a token "
                            "budget";
            return RAD_E_UNSUPPORTED;
        }
        if (!soft_budget_ok(image_tokens) ||
            (int64_t)image_tokens * cfg_.merge * cfg_.merge > cfg_.max_patches) {
            if (why) *why = "the image token budget must be one of 70, 140, 280, 560, 1120 and "
                            "fit one encoder pass (" + std::to_string(cfg_.max_patches) +
                            " patches)";
            return RAD_E_INVAL;
        }
    }
    Frame f;
    RAD_TRY(decode_image(bytes, n, cfg_.max_source_pixels, &f, why));
    if (cfg_.family == Family::Gemma4)
        return image_gemma4(f, image_tokens ? image_tokens : cfg_.image_soft_tokens, out, why);
    return image_from(f, out, why);
}

namespace {
struct SampleArgs { double fps; int32_t lo, hi; bool gemma4; };
std::vector<int64_t> sample_cb(const VideoInfo& info, const void* user) {
    const SampleArgs* a = static_cast<const SampleArgs*>(user);
    return a->gemma4 ? sample_frames_gemma4(info, a->fps, a->hi)
                     : sample_frames(info, a->fps, a->lo, a->hi);
}
}  /* namespace */

int Processor::video(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out,
                     std::string* why) const {
    if (cfg_.family == Family::Lfm2) {
        if (why) *why = "the LFM2 processor supports images only";
        return RAD_E_UNSUPPORTED;
    }
    Video v;
    const SampleArgs a{ cfg_.fps, cfg_.min_frames, cfg_.max_frames,
                        cfg_.family == Family::Gemma4 };
    RAD_TRY(decode_video(bytes, n, sample_cb, &a, cfg_.max_source_pixels, &v, why));
    return video_from(v, out, why);
}

/* ================================================================== audio */

/* A list a container flattened from a JSON array ("1 2 3"), or nothing when it is absent or null. */
static std::vector<double> read_list(const RadModelMeta& m, const char* key) {
    std::vector<double> v;
    const char* s = rad_meta_gets(&m, key, nullptr);
    if (!s || !*s) return v;
    std::istringstream is(s);
    double x;
    while (is >> x) v.push_back(x);
    return v;
}

int AudioConfig::from_meta(const RadModelMeta& m, std::string* why) {
    const char* mt = rad_meta_gets(&m, "audio_config.model_type", "");
    if (!mt || std::strcmp(mt, "gemma4_audio")) {
        if (why) *why = "the container states no audio tower this processor knows (audio_config."
                        "model_type gemma4_audio)";
        return RAD_E_NOTFOUND;
    }
    rate         = (int32_t)rad_meta_geti(&m, "preprocessor.sampling_rate", 16000);
    frame_length = (int32_t)rad_meta_geti(&m, "preprocessor.frame_length", 320);
    hop          = (int32_t)rad_meta_geti(&m, "preprocessor.hop_length", 160);
    n_mels       = (int32_t)rad_meta_geti(&m, "preprocessor.feature_size", 128);
    fft          = (int32_t)rad_meta_geti(&m, "preprocessor.fft_length", 0);
    if (fft <= 0) {
        fft = 1;
        while (fft < frame_length) fft *= 2;
    }
    if (rad_meta_geti(&m, "preprocessor.fft_overdrive", 0)) fft *= 2;
    min_freq    = rad_meta_getf(&m, "preprocessor.min_frequency", 0.0);
    max_freq    = rad_meta_getf(&m, "preprocessor.max_frequency", 8000.0);
    mel_floor   = rad_meta_getf(&m, "preprocessor.mel_floor", 1e-3);
    preemphasis = rad_meta_getf(&m, "preprocessor.preemphasis", 0.0);
    preemphasis_htk = rad_meta_geti(&m, "preprocessor.preemphasis_htk_flavor", 1) != 0;
    input_scale = rad_meta_getf(&m, "preprocessor.input_scale_factor", 1.0);
    bin_mean = read_list(m, "preprocessor.per_bin_mean");
    bin_std  = read_list(m, "preprocessor.per_bin_stddev");
    max_samples = 480000;
    tok_audio = (int32_t)rad_meta_geti(&m, "audio_token_id", -1);
    tok_start = (int32_t)rad_meta_geti(&m, "boa_token_id", -1);
    tok_end   = (int32_t)rad_meta_geti(&m, "eoa_token_id", rad_meta_geti(&m, "eoa_token_index", -1));
    /* DITHER IS NOISE: the reference adds Gaussian noise of this deviation to every sample, a
     * different draw each call, and no fixed input reproduces it. */
    if (rad_meta_getf(&m, "preprocessor.dither", 0.0) != 0.0) {
        if (why) *why = "the feature extractor dithers its input (preprocessor.dither), which no "
                        "deterministic front end reproduces";
        return RAD_E_UNSUPPORTED;
    }
    if (rate <= 0 || frame_length <= 1 || hop <= 0 || n_mels <= 0 || fft < frame_length ||
        !(max_freq > min_freq) || !(mel_floor > 0.0) || tok_audio < 0 || tok_start < 0 ||
        tok_end < 0 || (!bin_mean.empty() && (int64_t)bin_mean.size() != n_mels) ||
        (!bin_std.empty() && (int64_t)bin_std.size() != n_mels)) {
        if (why) *why = "the container's audio feature configuration is incomplete or "
                        "inconsistent (rate, frame and hop lengths, mel bins, frequencies, "
                        "audio_token_id, boa_token_id, eoa_token_id)";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int64_t AudioConfig::real_frames(int64_t n) const {
    /* Frame i's last sample is (frame_length / 2 zeros) + i * hop + frame_length, and it is real
     * when that is inside the clip: i * hop + frame_length < frame_length / 2 + n. */
    const int64_t room = n + frame_length / 2 - frame_length - 1;
    return room < 0 ? 0 : room / hop + 1;
}

namespace {

/* An in-place radix-2 complex FFT, n a power of two: the frame's real spectrum in double, as the
 * reference's numpy rfft computes it. */
void fft_inplace(std::vector<double>& re, std::vector<double>& im,
                 const std::vector<double>& tw_re, const std::vector<double>& tw_im,
                 const std::vector<int32_t>& brev) {
    const size_t n = re.size();
    for (size_t i = 1; i < n; ++i) {
        const size_t j = (size_t)brev[i];
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;   /* stage len's k-th twiddle is the table's k * step */
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const double wr = tw_re[k * step], wi = tw_im[k * step];
                const size_t a = i + k, b = i + k + len / 2;
                const double xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
    }
}

/* numpy.linspace's values: start + i * step, the last exactly `stop`. */
std::vector<double> linspace(double a, double b, int64_t n) {
    std::vector<double> v((size_t)n);
    const double step = n > 1 ? (b - a) / (double)(n - 1) : 0.0;
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = a + (double)i * step;
    if (n > 1) v[(size_t)n - 1] = b;
    return v;
}

}  /* namespace */

int Processor::init_audio(const AudioConfig& cfg) {
    if (cfg.merge <= 0 || cfg.max_frames <= 0 || cfg.max_frames % cfg.merge) return RAD_E_INVAL;
    acfg_ = cfg;
    /* np.hanning(frame_length + 1)[:-1], the periodic window, stored as float32. */
    hann_.resize((size_t)cfg.frame_length);
    for (int32_t i = 0; i < cfg.frame_length; ++i)
        hann_[(size_t)i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i /
                                                        (double)cfg.frame_length));
    /* transformers' mel_filter_bank(norm=None, mel_scale="htk"): triangles between consecutive
     * mel points, in Hz, over the FFT bins' frequencies. */
    const int64_t nb = cfg.fft / 2 + 1, nm = cfg.n_mels;
    auto hz2mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto mel2hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
    const std::vector<double> fft_f = linspace(0.0, (double)(cfg.rate / 2), nb);
    std::vector<double> filt = linspace(hz2mel(cfg.min_freq), hz2mel(cfg.max_freq), nm + 2);
    for (double& f : filt) f = mel2hz(f);
    mel_.assign((size_t)(nb * nm), 0.0);
    mel_lo_.assign((size_t)nm, 0);
    mel_hi_.assign((size_t)nm, 0);
    for (int64_t i = 0; i < nm; ++i) {
        int32_t lo = -1, hi = 0;
        for (int64_t k = 0; k < nb; ++k) {
            const double down = -(filt[(size_t)i] - fft_f[(size_t)k]) /
                                (filt[(size_t)i + 1] - filt[(size_t)i]);
            const double up = (filt[(size_t)i + 2] - fft_f[(size_t)k]) /
                              (filt[(size_t)i + 2] - filt[(size_t)i + 1]);
            const double w = std::max(0.0, std::min(down, up));
            mel_[(size_t)(i * nb + k)] = w;
            if (w != 0.0) { if (lo < 0) lo = (int32_t)k; hi = (int32_t)k + 1; }
        }
        /* A triangle covers a few bins; the sum skips the zeros, which add nothing to it. */
        mel_lo_[(size_t)i] = lo < 0 ? 0 : lo;
        mel_hi_[(size_t)i] = hi;
    }
    /* The FFT's twiddles, once: a butterfly that evaluates its own sin and cos makes the
     * transform cost a 30 s clip over 100 ms. */
    brev_.assign((size_t)cfg.fft, 0);
    for (int32_t i = 1, j = 0; i < cfg.fft; ++i) {
        int32_t bit = cfg.fft >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        brev_[(size_t)i] = j;
    }
    tw_re_.resize((size_t)cfg.fft / 2);
    tw_im_.resize((size_t)cfg.fft / 2);
    for (int32_t k = 0; k < cfg.fft / 2; ++k) {
        const double ang = -2.0 * M_PI * (double)k / (double)cfg.fft;
        tw_re_[(size_t)k] = std::cos(ang);
        tw_im_[(size_t)k] = std::sin(ang);
    }
    audio_on_ = true;
    return RAD_OK;
}

int Processor::audio_from(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                          std::string* why) const {
    if (!audio_on_) { if (why) *why = "this program declares no audio encoder"; return RAD_E_UNSUPPORTED; }
    const AudioConfig& c = acfg_;
    const int64_t n = std::min<int64_t>((int64_t)samples.size(), c.max_samples);
    const int64_t real = c.real_frames(n);
    const int64_t rows = (real + c.merge - 1) / c.merge;
    const int64_t frames = rows * c.merge;
    if (frames > c.max_frames) {
        if (why) *why = "the clip is more frames than one audio encoder pass holds";
        return RAD_E_INVAL;
    }
    auto it = std::make_shared<Item>();
    it->kind = Kind::Audio;
    it->grid_t = 1;
    it->grid_h = (int32_t)frames;
    it->grid_w = 1;
    it->patch_dim = c.n_mels;
    it->n_patch = frames;
    it->valid = (int32_t)real;
    it->rows_h = (int32_t)rows;
    it->rows_w = 1;
    it->seg_rows = (int32_t)rows;
    it->n_seg = rows > 0 ? 1 : 0;
    it->pixels.assign((size_t)(frames * c.n_mels), 0);

    const int64_t pad = c.frame_length / 2;
    const int64_t nb = c.fft / 2 + 1;
    std::vector<double> re((size_t)c.fft), im((size_t)c.fft), mag((size_t)nb);
    std::vector<float> fr((size_t)c.frame_length + 1);
    auto sample = [&](int64_t p) -> float {     /* the zero-led clip, zero past its end */
        const int64_t q = p - pad;
        return q >= 0 && q < n ? samples[(size_t)q] * (float)c.input_scale : 0.0f;
    };
    /* Frame i's windowed samples into dst, zero past the frame. */
    auto window = [&](int64_t i, double* dst) {
        for (int32_t j = 0; j <= c.frame_length; ++j) fr[(size_t)j] = sample(i * c.hop + j);
        for (int32_t j = 0; j < c.frame_length; ++j) {
            float v;
            if (c.preemphasis > 0.0) {
                const float pe = (float)c.preemphasis;
                v = c.preemphasis_htk ? (j == 0 ? fr[0] * (1.0f - pe) : fr[(size_t)j] - pe * fr[(size_t)j - 1])
                                      : fr[(size_t)j + 1] - pe * fr[(size_t)j];
            } else {
                v = fr[(size_t)j];
            }
            dst[j] = (double)(v * hann_[(size_t)j]);   /* the product in float32 */
        }
        for (int32_t j = c.frame_length; j < c.fft; ++j) dst[j] = 0.0;
    };
    /* Frame i's mel bins from its magnitudes. */
    auto emit = [&](int64_t i, const double* m) {
        uint16_t* dst = it->pixels.data() + i * c.n_mels;
        for (int32_t b = 0; b < c.n_mels; ++b) {
            double s = 0.0;
            const double* w = mel_.data() + (size_t)b * nb;
            for (int32_t k = mel_lo_[(size_t)b]; k < mel_hi_[(size_t)b]; ++k) s += m[k] * w[k];
            double v = std::log(s + c.mel_floor);
            if (!c.bin_mean.empty()) v -= c.bin_mean[(size_t)b];
            if (!c.bin_std.empty())  v /= c.bin_std[(size_t)b];
            dst[b] = rad_f32_to_bf16((float)v);
        }
    };
    /* TWO REAL FRAMES A COMPLEX TRANSFORM: frame a in the real part, frame b in the imaginary,
     * and the two spectra separate by symmetry -- A[k] = (Z[k] + conj Z[n-k]) / 2 and
     * B[k] = (Z[k] - conj Z[n-k]) / 2i. Half the transforms; the magnitudes agree with one
     * transform a frame to a few ulps of double. |X| is sqrt(re^2 + im^2), within an ulp of
     * numpy's hypot and far below the float32 the features are rounded to. */
    std::vector<double> mag2((size_t)nb);
    const int64_t N = c.fft;
    for (int64_t i = 0; i < real; i += 2) {
        const bool pair = i + 1 < real;
        window(i, re.data());
        if (pair) window(i + 1, im.data());
        else std::fill(im.begin(), im.end(), 0.0);
        fft_inplace(re, im, tw_re_, tw_im_, brev_);
        for (int64_t k = 0; k < nb; ++k) {
            const size_t a = (size_t)k, b = k ? (size_t)(N - k) : 0;
            const double ar = 0.5 * (re[a] + re[b]), ai = 0.5 * (im[a] - im[b]);
            const double br = 0.5 * (im[a] + im[b]), bi = -0.5 * (re[a] - re[b]);
            mag[a]  = std::sqrt(ar * ar + ai * ai);
            mag2[a] = std::sqrt(br * br + bi * bi);
        }
        emit(i, mag.data());
        if (pair) emit(i + 1, mag2.data());
    }
    content_key(*it, &it->key_hi, &it->key_lo);
    it->expand.push_back(c.tok_start);
    it->seg_at.push_back(1);
    it->expand.insert(it->expand.end(), (size_t)rows, c.tok_audio);
    it->expand.push_back(c.tok_end);
    /* A clip too short for one real frame is <boa><eoa> and nothing to encode. */
    if (rows == 0) it->encoded = true;
    *out = std::move(it);
    return RAD_OK;
}

int Processor::audio(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out,
                     std::string* why) const {
    if (!audio_on_) { if (why) *why = "this program declares no audio encoder"; return RAD_E_UNSUPPORTED; }
    std::vector<float> s;
    RAD_TRY(decode_audio(bytes, n, acfg_.rate, acfg_.max_samples, &s, why));
    return audio_from(s, out, why);
}

/* ================================================================== rotary layout */

void PromptMedia::layout() {
    rope.runs.clear();
    int64_t pos = 0, tok = 0;
    for (size_t pi = 0; pi < items.size(); ++pi) {
        const Placed& p = items[pi];
        const Item& it = *p.item;
        for (int32_t s = 0; s < it.n_seg; ++s) {
            const int64_t run_tok = (int64_t)p.tok + it.seg_at[(size_t)s];
            pos += run_tok - tok;            /* the text in front of the run counts up */
            RopeLayout::Run r;
            r.tok = (int32_t)run_tok;
            r.n = (int32_t)it.segment_rows(s);
            const int32_t rows_h = it.segments.empty() ? it.rows_h
                : it.segment_h(s) / it.merge_size();
            r.w = it.segments.empty() ? it.rows_w : it.segment_w(s) / it.merge_size();
            r.linear = it.linear_positions;
            r.base = (int32_t)pos;
            pos += r.linear ? r.n : std::max(rows_h, r.w);
            tok = run_tok + r.n;
            r.shift = (int32_t)(pos - tok);
            r.item = (int32_t)pi;
            r.row0 = (int32_t)it.row_at(s);
            rope.runs.push_back(r);
        }
    }
}

}  /* namespace mm */
}  /* namespace rad */
