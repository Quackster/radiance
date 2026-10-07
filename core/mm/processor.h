/* processor.h -- what an image or a video becomes before an encoder sees it (spec §11).
 *
 * Qwen-VL (Qwen2-VL through Qwen3.x) and LFM2-VL share decoding and encoder staging.
 * The Qwen family below fixes its arithmetic; LFM2-VL instead uses HWC patch values,
 * independent image tiles plus an optional thumbnail, and ordinary token positions.
 * In either family, what is family-specific is
 * the arithmetic, not the numbers: every size, rate and token id comes from the container, which
 * carries the checkpoint's config.json and its two preprocessor configs (rad-convert writes them
 * under `vision_config.*`, `preprocessor.*` and `video_preprocessor.*`). What the family fixes:
 *
 *   - A picture is resized so both sides are multiples of patch * merge and its area falls in the
 *     configured band, keeping the aspect ratio (smart_resize), with PIL's antialiased bicubic.
 *   - It is cut into patch x patch squares, `temporal` frames deep -- an image is repeated to fill
 *     the depth -- in MERGE-BLOCK ORDER: the merge x merge patches that become one output row are
 *     consecutive. Each patch row is (channel, frame, y, x), normalised by the configured mean and
 *     deviation, in bf16.
 *   - The prompt gets `<vision_start> pad*N <vision_end>` where the template put the media, N
 *     being the encoder's output rows. A video gets one such run per temporal group, each after
 *     the text `<T seconds>` giving the group's time -- which is how the model knows when a frame
 *     is, since the groups share one temporal rotary component.
 *   - Rotary positions are three components (temporal, height, width): text counts up in all
 *     three; a media run holds its start in the first and adds its row and column to the other
 *     two; the text after it continues from start + max(rows, cols). (RopeLayout below.)
 *
 * The encoder's input is patches plus each patch's (row, col) in its segment and the segment's
 * size, which is all a learned position table and a 2-D rotary embedding need; the processor
 * produces those too, so the core stages them without knowing what they are for.
 */
#pragma once
#include "mm/media.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct RadModelMeta;

namespace rad {
namespace mm {

enum class Kind { Image, Video, Audio };

/* THE PROCESSOR FAMILY, from the container's vision_config.model_type. The families differ in how a
 * picture is sized, what a patch value is, and how a video is sampled and spelled in the prompt;
 * the Item they produce and everything after it is shared.
 *
 *   QwenVL  smart_resize into a pixel band; patches (channel, frame, y, x) normalised by the
 *           configured mean and deviation; videos in temporal groups, each after a "<T seconds>".
 *   Gemma4  the largest size whose sides are multiples of patch * pooling and whose patch count
 *           fits the soft-token budget (budget * pooling^2), aspect ratio kept; patches (y, x,
 *           channel) as 2 * (u / 255) - 1, the reference model's own input scaling; a video is
 *           frames sampled at `fps` (at most `max_frames`, thinned evenly), each its own segment
 *           at the per-frame budget, each spelled <boi> placeholders <eoi> with no timestamp.
 *   Lfm2    RGB patches in merge-block order, independent variable-size tiles plus an optional
 *           thumbnail, with ordinary sequential positions through all image tokens. */
enum class Family { QwenVL, Gemma4, Lfm2 };

struct VisionConfig {
    Family family = Family::QwenVL;
    int32_t min_tiles = 2, max_tiles = 10, tile_size = 512;
    bool split_images = true, use_thumbnail = true;
    double max_pixels_tolerance = 2.0;
    int64_t cap_image_pixels = 1048576;
    int64_t max_segment_patches = 1024;
    int32_t patch = 0, temporal = 0, merge = 0;
    /* Gemma4: soft tokens (output rows) an image, and a video frame, may take. One of 70, 140,
     * 280, 560, 1120 -- the budgets the model was trained at. */
    int32_t image_soft_tokens = 0, video_soft_tokens = 0;
    int64_t image_min_pixels = 0, image_max_pixels = 0;
    int64_t video_min_pixels = 0, video_max_pixels = 0;
    double  fps = 2.0;
    int32_t min_frames = 4, max_frames = 768;
    /* The per-frame ceiling on a video's resolution, in output rows: a clip that samples few
     * frames does not spend the whole video budget on near-native frames, so the cost of a video
     * grows with its length (the reference processor's per-frame cap). 0 turns the cap off. */
    int32_t max_frame_rows = 768;
    float   mean[3] = { 0.5f, 0.5f, 0.5f };
    float   stdv[3] = { 0.5f, 0.5f, 0.5f };
    int32_t tok_image = -1, tok_video = -1, tok_start = -1, tok_end = -1;

    /* THE DEPLOYMENT'S BOUNDS, not the checkpoint's. `max_patches` is the encoder's pass size: an
     * image is one segment and a segment is never split, so no image may resize past it, and the
     * image band's ceiling is lowered to fit. `max_source_pixels` bounds what the decoder is asked
     * to hold before any resize. */
    int64_t max_patches = 0;
    int64_t max_source_pixels = (int64_t)1 << 26;
    /* A whole video's patch budget, across all its passes. */
    int64_t max_video_patches = 0;

    /* The pixel bands are non-empty and the frame counts and rate positive. RAD_OK, or
     * RAD_E_INVAL with the first one that is not in `why`. from_meta runs it on what the
     * container states, and the engine again after the command line has replaced some of it. */
    int check(std::string* why) const;

    int64_t patch_dim() const { return 3LL * temporal * patch * patch; }
    int32_t factor() const { return patch * merge; }

    /* Read from the container. Refuses a container that states no vision tower; the defaults for
     * the processor-only keys are the family's, used only when the container omits them. */
    int from_meta(const RadModelMeta& m, std::string* why);
};

/* Retained encoder output on the rank that supplies multimodal embeddings. A
 * bounded pool owns the allocation; the last shared media owner releases its
 * spans. Spans are in logical row order, though fragmentation may place their
 * storage apart. No device allocation is performed by an Item. */
struct DeviceRows {
    struct Span { int64_t row = 0, count = 0; void* data = nullptr; };
    int device = -1;
    int64_t rows = 0, width = 0;
    std::vector<Span> spans;
    std::function<void()> release;
    DeviceRows() = default;
    DeviceRows(const DeviceRows&) = delete;
    DeviceRows& operator=(const DeviceRows&) = delete;
    ~DeviceRows() { if (release) release(); }
};

/* WHAT AN AUDIO CLIP BECOMES: Gemma 4's feature extractor (Gemma4AudioFeatureExtractor), read
 * from the container's `preprocessor.*` (the checkpoint's preprocessor_config.json) and its
 * audio_config. The samples are mono at `rate`, cut at `max_samples`; `frame_length / 2` zeros go
 * in front; frame i is samples [i * hop, i * hop + frame_length) of that, pre-emphasised when the
 * config says so, times a periodic Hann window, through a `fft`-point real FFT; its magnitudes go
 * through an HTK mel bank (no area normalisation) and log(x + mel_floor), optionally standardised
 * per bin. A frame is real when its last sample is a real one -- the reference's mask -- so a clip
 * of n samples has floor((n - frame_length / 2 - 1) / hop) + 1 real frames and the reference's
 * output keeps ceil(real / merge) rows of it: that many placeholders, between <boa> and <eoa>.
 * The staged patches are the real frames then zero frames to a whole number of rows -- equal, for
 * every row the reference keeps, to the frames past the end the reference zeroes. */
struct AudioConfig {
    int32_t rate = 0, frame_length = 0, hop = 0, fft = 0, n_mels = 0;
    double  min_freq = 0.0, max_freq = 0.0, mel_floor = 0.0;
    double  preemphasis = 0.0, input_scale = 1.0;
    bool    preemphasis_htk = true;
    std::vector<double> bin_mean, bin_std;      /* empty: no per-bin standardisation */
    /* The longest clip in samples: Gemma 4's extractor truncates at 480000 (30 s at 16 kHz). */
    int64_t max_samples = 0;
    int32_t tok_audio = -1, tok_start = -1, tok_end = -1;

    /* THE ENCODER'S: frames an output row, and the most frames a pass carries. */
    int32_t merge = 0;
    int64_t max_frames = 0;

    /* Read from the container; refuses one that states no Gemma 4 audio tower. */
    int from_meta(const RadModelMeta& m, std::string* why);
    /* The real frames of an n-sample clip. */
    int64_t real_frames(int64_t n_samples) const;
};

/* One media part after processing: its patches, its place in the prompt's token stream, and --
 * once the encoder has run -- its rows. Shared between the request that owns it and the engine
 * thread that encodes it; the engine writes `embd` once and nothing reads it before then. */
struct Item {
    Kind    kind = Kind::Image;
    int32_t grid_t = 0, grid_h = 0, grid_w = 0;   /* in patches; grid_t temporal groups */
    int64_t n_patch = 0;
    int64_t patch_dim = 0;
    std::vector<uint16_t> pixels;                 /* [n_patch, patch_dim] bf16 */

    /* The encoder's output: n_seg segments of seg_rows rows, one segment per temporal group.
     * `rows_w` is a segment's width in output rows, which is where its rotary grid wraps. */
    int32_t n_seg = 0, seg_rows = 0, rows_w = 0, rows_h = 0;
    /* Optional variable-size segments, used by tiled images plus their thumbnail.
     * Empty keeps uniform image/video segments or a one-dimensional audio clip. */
    struct Segment { int32_t h = 0, w = 0; int64_t patch0 = 0; };
    std::vector<Segment> segments;
    bool linear_positions = false;
    int32_t segment_h(int s) const { return segments.empty() ? grid_h : segments[s].h; }
    int32_t segment_w(int s) const { return segments.empty() ? grid_w : segments[s].w; }
    int32_t merge_size() const { return grid_h / rows_h; }
    int64_t patch_at(int s) const {
        return segments.empty() ? (int64_t)s * grid_h * grid_w
             : s == n_seg ? n_patch : segments[s].patch0;
    }
    int64_t row_at(int s) const { return segments.empty() ? (int64_t)s * seg_rows
             : patch_at(s) / (merge_size() * merge_size()); }
    int64_t segment_patches(int s) const { return patch_at(s + 1) - patch_at(s); }
    int64_t segment_rows(int s) const { return row_at(s + 1) - row_at(s); }
    int64_t n_rows() const { return row_at(n_seg); }
    /* An audio clip's REAL feature frames; the rest of its `grid_h` are the zero frames that make
     * it whole output rows. 0 for a picture or a video. */
    int32_t valid = 0;

    /* THE TOKENS THAT REPLACE THE MEDIA MARKER, and where in them each segment's run of
     * placeholders starts. Placeholders are the checkpoint's own pad ids, not an arbitrary one:
     * an n-gram embedding hashes them like any other id. */
    std::vector<int32_t> expand;
    std::vector<int32_t> seg_at;

    /* A CONTENT KEY over the patches and the grid. It is taken BEFORE the encoder runs, so a
     * prefix-cache hit over this item skips the encoder as well as the prefill: the patches are
     * what the encoder is a function of, and they already reflect every preprocessing choice. */
    uint64_t key_hi = 0, key_lo = 0;

    /* [n_rows, n_embd] bf16, written by the engine after the encoder pass.
     * Real devices prefer the accounted device pool; host execution, pool
     * pressure and explicit diagnostic dumps retain the host representation. */
    std::vector<uint16_t> embd;
    std::shared_ptr<DeviceRows> device_rows;
    bool encoded = false;
    /* Handed to the engine's encoder queue and not yet encoded -- the scheduler's thread only. */
    bool queued = false;

    /* The (row, col, rows, cols) of each patch in its segment, for the encoder: component-major,
     * four planes of `count` entries, patches [first, first + count). */
    void coords(int64_t first, int64_t count, int32_t* planes) const;
};

/* THE ROTARY LAYOUT OF ONE PROMPT: which tokens are media and where each run's grid sits. Built
 * once when the prompt is tokenised; `at` is what the batch builder asks for every row of every
 * pass, so it is a binary search over runs rather than an array the length of the prompt. */
struct RopeLayout {
    struct Run {
        int32_t tok = 0;        /* first placeholder token of the run */
        int32_t n = 0;          /* its length: rows_h * rows_w */
        int32_t w = 0;          /* its width in rows */
        int32_t base = 0;       /* the rotary position of its first token */
        int32_t shift = 0;      /* rotary minus token index for the text AFTER the run */
        int32_t item = 0;       /* which of PromptMedia::items the run belongs to */
        bool linear = false;   /* ordinary one-dimensional positions through media */
        int32_t row0 = 0;       /* the item's encoder row that its first token takes */
    };
    std::vector<Run> runs;      /* ascending */

    bool empty() const { return runs.empty(); }

    /* The last run starting at or before token i, or null. Inline and dependency-free, because
     * the batch builder asks it for every row of every pass. */
    const Run* last_at(int64_t i) const {
        size_t lo = 0, hi = runs.size();
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if ((int64_t)runs[mid].tok <= i) lo = mid + 1; else hi = mid;
        }
        return lo == 0 ? nullptr : &runs[lo - 1];
    }
    /* (t, h, w) of token index i. Past the prompt, text continues from the last run's shift. */
    void at(int64_t i, int32_t out[3]) const {
        const Run* r = last_at(i);
        if (!r || r->linear) { out[0] = out[1] = out[2] = (int32_t)i; return; }
        const int64_t k = i - r->tok;
        if (k < r->n) {
            out[0] = r->base;
            out[1] = r->base + (int32_t)(k / r->w);
            out[2] = r->base + (int32_t)(k % r->w);
            return;
        }
        out[0] = out[1] = out[2] = (int32_t)(i + r->shift);
    }
    /* Rotary minus index for text at or after token i -- the one number a decode row needs. */
    int32_t shift_at(int64_t i) const {
        const Run* r = last_at(i);
        return r ? r->shift : 0;
    }
    /* The run holding token i, or null when i is text. */
    const Run* run_at(int64_t i) const {
        const Run* r = last_at(i);
        return r && i - r->tok < r->n ? r : nullptr;
    }
};

/* THE MEDIA A PROMPT CARRIES, in prompt order, with where each item's expansion begins. */
struct Placed {
    std::shared_ptr<Item> item;
    int32_t               tok = 0;   /* index of item->expand[0] in the prompt */
};

struct PromptMedia {
    std::vector<Placed> items;
    RopeLayout          rope;
    /* Build `rope` from `items`, which must be placed and in prompt order. */
    void layout();
};

class Processor {
public:
    /* `encode_text` tokenises the timestamp text between a video's groups: plain text, no
     * special tokens added and none parsed. `encode_special` resolves LFM2 image markers. */
    int init(const VisionConfig& cfg,
             std::function<std::vector<int32_t>(const std::string&)> encode_text,
             std::function<std::vector<int32_t>(const std::string&)> encode_special = {});
    const VisionConfig& config() const { return cfg_; }

    /* `image_tokens` is a Gemma 4 soft-token budget for this image (0: the configured one); a
     * processor that sizes images by pixels refuses a non-zero one. */
    int image(const uint8_t* bytes, size_t n, int32_t image_tokens, std::shared_ptr<Item>* out,
              std::string* why) const;
    int video(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out, std::string* why) const;

    /* The pieces, exposed for the test suite: an already-decoded picture or frame list. */
    int image_from(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int video_from(const Video& v, std::shared_ptr<Item>* out, std::string* why) const;

    /* AUDIO, when the program declared an audio encoder: `merge` and `max_frames` are its. */
    int  init_audio(const AudioConfig& cfg);
    bool has_audio() const { return audio_on_; }
    const AudioConfig& audio_config() const { return acfg_; }
    int audio(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out, std::string* why) const;
    /* ...and from samples already decoded (mono at the configured rate), for the test suite. */
    int audio_from(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                   std::string* why) const;

private:
    int image_lfm2(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int image_gemma4(const Frame& f, int32_t soft, std::shared_ptr<Item>* out,
                     std::string* why) const;
    int video_gemma4(const Video& v, std::shared_ptr<Item>* out, std::string* why) const;

    VisionConfig cfg_{};
    int32_t tok_thumbnail_ = -1;
    std::vector<int32_t> tile_tokens_;
    AudioConfig  acfg_{};
    bool         audio_on_ = false;
    std::vector<float>  hann_;          /* the window, float32 as the reference stores it */
    std::vector<double> mel_;           /* [n_mels, fft/2 + 1]: filter b's weight on bin k */
    std::vector<int32_t> mel_lo_, mel_hi_;  /* filter b's nonzero bins, [lo, hi) */
    std::vector<double> tw_re_, tw_im_;     /* exp(-2 pi i k / fft), k < fft/2 */
    std::vector<int32_t> brev_;             /* the FFT's bit-reversal permutation */
    std::function<std::vector<int32_t>(const std::string&)> encode_text_;
};

/* smart_resize, the image form: (height, width) -> multiples of `factor` in [min, max] pixels. */
int smart_resize_image(int64_t h, int64_t w, int32_t factor, int64_t min_px, int64_t max_px,
                       int64_t* oh, int64_t* ow, std::string* why);
/* ...and the video form, which counts the frames into the budget. */
int smart_resize_video(int64_t t, int64_t h, int64_t w, int32_t tfactor, int32_t factor,
                       int64_t min_px, int64_t max_px, int64_t* oh, int64_t* ow, std::string* why);
/* Which frames a video keeps. */
std::vector<int64_t> sample_frames(const VideoInfo& info, double fps, int32_t min_frames,
                                   int32_t max_frames);
/* ...the Gemma 4 rule: one frame each 1 / fps seconds, thinned evenly to `max_frames`. */
std::vector<int64_t> sample_frames_gemma4(const VideoInfo& info, double fps, int32_t max_frames);
/* Gemma 4's target size: the largest (h, w), both multiples of `side`, whose patch count stays
 * within `max_patches`, aspect ratio kept (get_aspect_ratio_preserving_size). */
int gemma4_size(int64_t h, int64_t w, int32_t patch, int32_t pool, int64_t max_patches,
                int64_t* oh, int64_t* ow, std::string* why);
/* PIL's antialiased bicubic resample of packed RGB, bit for bit: fixed-point coefficients, a
 * horizontal pass into an 8-bit intermediate and then a vertical one. */
void resize_bicubic(const Frame& in, int32_t w, int32_t h, Frame* out);

}  /* namespace mm */
}  /* namespace rad */
