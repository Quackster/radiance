/* media.h -- turning the bytes of an image or a video into RGB frames (spec §11).
 *
 * ONE DECODER FOR BOTH, and it is libavformat/libavcodec. An image is a one-frame stream to it, so
 * PNG, JPEG, WebP, GIF, BMP and TIFF arrive through the same demuxer probe and the same decode loop
 * a video takes, and there is one colour conversion (libswscale) rather than one per format. A
 * build without the libraries has no decoder at all: every call here returns RAD_E_UNSUPPORTED
 * with the reason, and the server refuses a media part by name rather than guessing at its bytes.
 *
 * THE BYTES COME FROM THE REQUEST, never from a URL. The server accepts `data:` URLs only, so
 * nothing here opens a file or a network stream: the demuxer reads a memory buffer through its own
 * I/O context, and the protocol whitelist is empty.
 *
 * WHAT A FRAME IS: packed 8-bit RGB, row-major, no padding. An alpha channel is DROPPED, not
 * composited -- the reference processor converts with PIL's convert("RGB"), which does the same,
 * so a transparent PNG reaches the encoder with the colours under its alpha.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace mm {

struct Frame {
    int32_t              w = 0, h = 0;
    std::vector<uint8_t> rgb;          /* [h, w, 3] */
};

/* Is there a decoder in this build. */
bool decoder_available();

/* The first frame of an image (or of anything with a video stream). Refuses more than
 * `max_pixels` in the SOURCE, which is what bounds the memory a request can make the server
 * allocate before any resize happens. */
int decode_image(const uint8_t* data, size_t n, int64_t max_pixels, Frame* out, std::string* why);

/* HOW A VIDEO IS SAMPLED, and it happens INSIDE the decode rather than after it: a minute of 1080p
 * is 10 GB of RGB, so frames that are not sampled are never converted or kept. The choice of which
 * frames is the processor's (it depends on the stream's length and rate, which only the decoder
 * knows), so it is a callback: given the stream's frame count and rate, return the frame indices to
 * keep, ascending. */
struct VideoInfo {
    int64_t n_frames = 0;    /* frames in the stream; counted by demuxing when the header omits it */
    double  fps = 0.0;       /* the stream's average rate */
    int32_t w = 0, h = 0;
};
typedef std::vector<int64_t> (*SampleFn)(const VideoInfo& info, const void* user);

struct Video {
    VideoInfo            info;
    std::vector<int64_t> index;        /* the frame each of `frames` is */
    std::vector<Frame>   frames;
};

int decode_video(const uint8_t* data, size_t n, SampleFn sample, const void* user,
                 int64_t max_pixels, Video* out, std::string* why);

/* AN AUDIO CLIP as mono float samples at `rate` Hz: the first audio stream, resampled with
 * libswresample at the source's channel layout and then averaged across channels -- the plain
 * mean every reference loader (librosa, torchaudio's to-mono) takes, where swresample's own
 * downmix would weight a stereo pair by 0.707. At most `max_samples` are kept: a longer clip is
 * cut there, as the reference feature extractor truncates. */
int decode_audio(const uint8_t* data, size_t n, int32_t rate, int64_t max_samples,
                 std::vector<float>* out, std::string* why);

}  /* namespace mm */
}  /* namespace rad */
