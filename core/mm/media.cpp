/* media.cpp -- image and video decoding through libavformat / libavcodec / libswscale. */
#include "mm/media.h"

#include "rad_internal.h"

#include <algorithm>
#include <cstring>

#ifdef RAD_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

namespace rad {
namespace mm {

#ifndef RAD_HAVE_FFMPEG

bool decoder_available() { return false; }

static int no_decoder(std::string* why) {
    if (why) *why = "this build has no media decoder: it was configured without libavformat / "
                    "libavcodec / libswscale (RAD_WITH_FFMPEG)";
    return RAD_E_UNSUPPORTED;
}

int decode_image(const uint8_t*, size_t, int64_t, Frame*, std::string* why) {
    return no_decoder(why);
}
int decode_video(const uint8_t*, size_t, SampleFn, const void*, int64_t, Video*,
                 std::string* why) {
    return no_decoder(why);
}
int decode_audio(const uint8_t*, size_t, int32_t, int64_t, std::vector<float>*,
                 std::string* why) {
    return no_decoder(why);
}

#else

bool decoder_available() { return true; }

namespace {

/* THE DEMUXERS A REQUEST MAY REACH, and nothing else. A container format that can name another
 * file -- a playlist, a concat script, an image sequence pattern -- would otherwise let a request
 * make the server open a path of its choosing, so the probe only ever considers these, and the
 * protocol list below is empty so none of them can open anything either. */
constexpr const char* kFormats =
    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts,flv,ogg,gif,apng,image2pipe,png_pipe,"
    "jpeg_pipe,jpegls_pipe,jpegxl_pipe,webp_pipe,bmp_pipe,tiff_pipe,qoi_pipe,pgm_pipe,ppm_pipe,"
    "pbm_pipe,pam_pipe,gif_pipe,avif,heif";

struct MemIO {
    const uint8_t* p = nullptr;
    size_t         n = 0;
    size_t         at = 0;
};

int mem_read(void* opaque, uint8_t* buf, int size) {
    MemIO* m = static_cast<MemIO*>(opaque);
    const size_t left = m->n - m->at;
    if (left == 0) return AVERROR_EOF;
    const size_t k = std::min(left, (size_t)size);
    std::memcpy(buf, m->p + m->at, k);
    m->at += k;
    return (int)k;
}

int64_t mem_seek(void* opaque, int64_t off, int whence) {
    MemIO* m = static_cast<MemIO*>(opaque);
    if (whence & AVSEEK_SIZE) return (int64_t)m->n;
    int64_t base = 0;
    switch (whence & ~AVSEEK_FORCE) {
        case SEEK_SET: base = 0; break;
        case SEEK_CUR: base = (int64_t)m->at; break;
        case SEEK_END: base = (int64_t)m->n; break;
        default: return AVERROR(EINVAL);
    }
    const int64_t to = base + off;
    if (to < 0 || to > (int64_t)m->n) return AVERROR(EINVAL);
    m->at = (size_t)to;
    return to;
}

std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, b, sizeof b);
    return b;
}

/* One opened input over a memory buffer: the demuxer, its best video stream and a decoder for it.
 * Every libav object is released in the destructor, so an early return anywhere leaks nothing. */
struct Input {
    MemIO            io;
    AVIOContext*     avio = nullptr;
    AVFormatContext* fmt = nullptr;
    AVCodecContext*  dec = nullptr;
    int              stream = -1;

    ~Input() {
        if (dec) avcodec_free_context(&dec);
        if (fmt) avformat_close_input(&fmt);
        if (avio) {
            av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
    }

    int open(const uint8_t* data, size_t n, bool want_decoder, std::string* why) {
        io.p = data; io.n = n; io.at = 0;
        constexpr int kBuf = 64 * 1024;
        uint8_t* buf = static_cast<uint8_t*>(av_malloc(kBuf));
        if (!buf) { *why = "out of memory"; return RAD_E_NOMEM; }
        avio = avio_alloc_context(buf, kBuf, 0, &io, mem_read, nullptr, mem_seek);
        if (!avio) { av_free(buf); *why = "out of memory"; return RAD_E_NOMEM; }
        fmt = avformat_alloc_context();
        if (!fmt) { *why = "out of memory"; return RAD_E_NOMEM; }
        fmt->pb = avio;
        fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "format_whitelist", kFormats, 0);
        av_dict_set(&opts, "protocol_whitelist", "", 0);
        int e = avformat_open_input(&fmt, nullptr, nullptr, &opts);
        av_dict_free(&opts);
        if (e < 0) {
            fmt = nullptr;   /* freed by a failed open */
            *why = "not a recognised image or video format (" + averr(e) + ")";
            return RAD_E_INVAL;
        }
        e = avformat_find_stream_info(fmt, nullptr);
        if (e < 0) { *why = "could not read the stream headers (" + averr(e) + ")"; return RAD_E_INVAL; }
        const AVCodec* codec = nullptr;
        stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        if (stream < 0 || !codec) { *why = "no picture in it: the media holds no video stream"; return RAD_E_INVAL; }
        if (!want_decoder) return RAD_OK;
        dec = avcodec_alloc_context3(codec);
        if (!dec) { *why = "out of memory"; return RAD_E_NOMEM; }
        e = avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar);
        if (e < 0) { *why = "decoder setup failed (" + averr(e) + ")"; return RAD_E_INVAL; }
        /* A few threads a decode: this runs on a server worker, one per request in flight, so
         * handing each one every core would oversubscribe the machine the engine runs on. */
        dec->thread_count = 4;
        e = avcodec_open2(dec, codec, nullptr);
        if (e < 0) { *why = std::string("no decoder for ") + codec->name + " (" + averr(e) + ")"; return RAD_E_UNSUPPORTED; }
        return RAD_OK;
    }
};

/* One decoded frame to packed RGB24 at its own size. The YUV coefficients and range are the
 * frame's own, so a BT.709 video and a full-range JPEG both come out with the colours they were
 * encoded with.
 *
 * THE SCALER FLAGS ARE THE REFERENCE PIPELINE'S: transformers decodes video through PyAV
 * (`to_ndarray(format="rgb24")`, its default backend), which converts with plain SWS_BILINEAR.
 * The flags decide how chroma is upsampled, and at a sharp colour edge two choices differ by up to
 * 70 levels: measured on an EmbeddingGemma 2 clip, SWS_BICUBIC | ACCURATE_RND | FULL_CHR_H_INT
 * left the embedding at cosine 0.9995 against the reference, these at 0.99998 with every patch
 * within one level. */
int to_rgb(const AVFrame* f, SwsContext** sws, Frame* out, std::string* why) {
    const AVPixelFormat src = (AVPixelFormat)f->format;
    *sws = sws_getCachedContext(*sws, f->width, f->height, src, f->width, f->height,
                                AV_PIX_FMT_RGB24,
                                SWS_BILINEAR, nullptr,
                                nullptr, nullptr);
    if (!*sws) { *why = "no colour conversion from this pixel format"; return RAD_E_UNSUPPORTED; }
    int cs = f->colorspace;
    if (cs == AVCOL_SPC_UNSPECIFIED || cs == AVCOL_SPC_RGB) cs = AVCOL_SPC_BT470BG;
    const int* coef = sws_getCoefficients(cs);
    const int src_full = f->color_range == AVCOL_RANGE_JPEG ||
                         src == AV_PIX_FMT_YUVJ420P || src == AV_PIX_FMT_YUVJ422P ||
                         src == AV_PIX_FMT_YUVJ444P || src == AV_PIX_FMT_YUVJ440P;
    sws_setColorspaceDetails(*sws, coef, src_full, sws_getCoefficients(SWS_CS_DEFAULT), 1, 0,
                             1 << 16, 1 << 16);
    out->w = f->width;
    out->h = f->height;
    out->rgb.resize((size_t)f->width * (size_t)f->height * 3);
    uint8_t* dst[4] = { out->rgb.data(), nullptr, nullptr, nullptr };
    int dst_ls[4] = { f->width * 3, 0, 0, 0 };
    const int e = sws_scale(*sws, f->data, f->linesize, 0, f->height, dst, dst_ls);
    if (e != f->height) { *why = "colour conversion failed"; return RAD_E_INVAL; }
    return RAD_OK;
}

/* THE DECODE LOOP, for both kinds: every frame in presentation order, `keep(i, frame)` told the
 * index of each and whether to stop.
 *
 * AN EMPTY PACKET REPEATS THE FRAME BEFORE IT: it is how Ogg Theora stores a frame that does not
 * change. Sent to a decoder, an empty packet would read as the end of the stream and the next one
 * would be refused; it is passed to `keep` as another copy of the last frame instead, so the
 * frame indices and the stream's length are the ones the frame count promised. */
template <class Keep>
int decode_frames(Input& in, Keep&& keep, std::string* why) {
    AVPacket* pkt = av_packet_alloc();
    AVFrame*  frm = av_frame_alloc();
    AVFrame*  last = av_frame_alloc();
    if (!pkt || !frm || !last) {
        av_packet_free(&pkt);
        av_frame_free(&frm);
        av_frame_free(&last);
        *why = "out of memory";
        return RAD_E_NOMEM;
    }
    int64_t idx = 0;
    int st = RAD_OK;
    bool done = false, flushing = false;
    while (!done && st >= 0) {
        if (!flushing) {
            const int r = av_read_frame(in.fmt, pkt);
            if (r == AVERROR_EOF) {
                flushing = true;
                avcodec_send_packet(in.dec, nullptr);
            } else if (r < 0) {
                *why = "demuxing failed (" + averr(r) + ")";
                st = RAD_E_INVAL;
                break;
            } else {
                if (pkt->stream_index != in.stream) { av_packet_unref(pkt); continue; }
                if (pkt->size == 0) {
                    av_packet_unref(pkt);
                    if (!last->buf[0]) continue;
                    const int k = keep(idx++, last);
                    if (k < 0) { st = k; break; }
                    if (k > 0) done = true;
                    continue;
                }
                const int s = avcodec_send_packet(in.dec, pkt);
                av_packet_unref(pkt);
                if (s < 0 && s != AVERROR(EAGAIN)) {
                    *why = "decoding failed (" + averr(s) + ")";
                    st = RAD_E_INVAL;
                    break;
                }
            }
        }
        for (;;) {
            const int r = avcodec_receive_frame(in.dec, frm);
            if (r == AVERROR(EAGAIN)) break;
            if (r == AVERROR_EOF) { done = true; break; }
            if (r < 0) {
                *why = "decoding failed (" + averr(r) + ")";
                st = RAD_E_INVAL;
                break;
            }
            const int k = keep(idx++, frm);
            av_frame_unref(last);
            av_frame_move_ref(last, frm);
            if (k < 0) { st = k; break; }
            if (k > 0) { done = true; break; }
        }
    }
    av_packet_free(&pkt);
    av_frame_free(&frm);
    av_frame_free(&last);
    return st;
}

}  /* namespace */

int decode_image(const uint8_t* data, size_t n, int64_t max_pixels, Frame* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    if (!data || n == 0) { *why = "empty"; return RAD_E_INVAL; }
    Input in;
    RAD_TRY(in.open(data, n, true, why));
    const AVCodecParameters* cp = in.fmt->streams[in.stream]->codecpar;
    if (cp->width <= 0 || cp->height <= 0) { *why = "the image has no size"; return RAD_E_INVAL; }
    if ((int64_t)cp->width * cp->height > max_pixels) {
        *why = "the image is " + std::to_string(cp->width) + "x" + std::to_string(cp->height) +
               ", over the server's " + std::to_string(max_pixels) + "-pixel limit for a source";
        return RAD_E_INVAL;
    }
    SwsContext* sws = nullptr;
    bool got = false;
    const int s = decode_frames(in, [&](int64_t, const AVFrame* f) -> int {
        const int c = to_rgb(f, &sws, out, why);
        if (c < 0) return c;
        got = true;
        return 1;
    }, why);
    sws_freeContext(sws);
    if (s < 0) return s;
    if (!got) { *why = "the image decoded to no frame"; return RAD_E_INVAL; }
    return RAD_OK;
}

int decode_video(const uint8_t* data, size_t n, SampleFn sample, const void* user,
                 int64_t max_pixels, Video* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    if (!data || n == 0) { *why = "empty"; return RAD_E_INVAL; }

    /* THE FRAME COUNT FIRST, because which frames to keep depends on it. Many containers state it;
     * those that do not are demuxed once without decoding, which counts packets -- one a frame for
     * every codec a request is going to send -- at the cost of reading the bytes twice. */
    VideoInfo info;
    {
        Input probe;
        RAD_TRY(probe.open(data, n, false, why));
        const AVStream* st = probe.fmt->streams[probe.stream];
        info.w = st->codecpar->width;
        info.h = st->codecpar->height;
        AVRational r = st->avg_frame_rate;
        if (r.num <= 0 || r.den <= 0) r = st->r_frame_rate;
        info.fps = (r.num > 0 && r.den > 0) ? av_q2d(r) : 0.0;
        info.n_frames = st->nb_frames;
        if (info.n_frames <= 0) {
            AVPacket* pkt = av_packet_alloc();
            if (!pkt) { *why = "out of memory"; return RAD_E_NOMEM; }
            int64_t c = 0;
            while (av_read_frame(probe.fmt, pkt) >= 0) {
                if (pkt->stream_index == probe.stream) ++c;
                av_packet_unref(pkt);
            }
            av_packet_free(&pkt);
            info.n_frames = c;
        }
    }
    if (info.w <= 0 || info.h <= 0) { *why = "the video has no size"; return RAD_E_INVAL; }
    if ((int64_t)info.w * info.h > max_pixels) {
        *why = "the video is " + std::to_string(info.w) + "x" + std::to_string(info.h) +
               ", over the server's " + std::to_string(max_pixels) + "-pixel limit for a frame";
        return RAD_E_INVAL;
    }
    if (info.n_frames <= 0) { *why = "the video holds no frames"; return RAD_E_INVAL; }

    out->info = info;
    out->index = sample(info, user);
    if (out->index.empty()) { *why = "no frames were sampled"; return RAD_E_INVAL; }
    out->frames.assign(out->index.size(), Frame{});

    Input in;
    RAD_TRY(in.open(data, n, true, why));
    SwsContext* sws = nullptr;
    size_t next = 0;
    int64_t last_kept = -1;
    const int s = decode_frames(in, [&](int64_t i, const AVFrame* f) -> int {
        if ((int64_t)f->width * f->height > max_pixels) {
            *why = "a frame is larger than the server's pixel limit";
            return RAD_E_INVAL;
        }
        /* The same frame may be sampled twice when the stream is shorter than its own header
         * says; each slot that names this index takes a copy. */
        bool converted = false;
        while (next < out->index.size() && out->index[next] == i) {
            if (!converted) {
                const int c = to_rgb(f, &sws, &out->frames[next], why);
                if (c < 0) return c;
                converted = true;
            } else {
                out->frames[next] = out->frames[next - 1];
            }
            last_kept = (int64_t)next;
            ++next;
        }
        return next >= out->index.size() ? 1 : 0;
    }, why);
    sws_freeContext(sws);
    if (s < 0) return s;
    if (last_kept < 0) { *why = "the video decoded to no frame"; return RAD_E_INVAL; }
    /* A STREAM SHORTER THAN ITS HEADER leaves the last sampled indices unreached. They take the
     * last frame that was decoded, which is what a reader that clamps an index past the end
     * returns. */
    for (size_t k = next; k < out->index.size(); ++k) out->frames[k] = out->frames[(size_t)last_kept];
    return RAD_OK;
}

/* ================================================================== audio */

namespace {

/* THE AUDIO DEMUXERS A REQUEST MAY REACH, chosen as the video list is: formats that hold one
 * stream of samples and cannot name another file. */
constexpr const char* kAudioFormats = "wav,w64,mp3,flac,ogg,mov,mp4,m4a,matroska,webm,aac,aiff";

struct AudioIn {
    MemIO            io;
    AVIOContext*     avio = nullptr;
    AVFormatContext* fmt = nullptr;
    AVCodecContext*  dec = nullptr;
    SwrContext*      swr = nullptr;
    int              stream = -1;

    ~AudioIn() {
        if (swr) swr_free(&swr);
        if (dec) avcodec_free_context(&dec);
        if (fmt) avformat_close_input(&fmt);
        if (avio) {
            av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
    }

    int open(const uint8_t* data, size_t n, std::string* why) {
        io.p = data; io.n = n; io.at = 0;
        constexpr int kBuf = 64 * 1024;
        uint8_t* buf = static_cast<uint8_t*>(av_malloc(kBuf));
        if (!buf) { *why = "out of memory"; return RAD_E_NOMEM; }
        avio = avio_alloc_context(buf, kBuf, 0, &io, mem_read, nullptr, mem_seek);
        if (!avio) { av_free(buf); *why = "out of memory"; return RAD_E_NOMEM; }
        fmt = avformat_alloc_context();
        if (!fmt) { *why = "out of memory"; return RAD_E_NOMEM; }
        fmt->pb = avio;
        fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "format_whitelist", kAudioFormats, 0);
        av_dict_set(&opts, "protocol_whitelist", "", 0);
        int e = avformat_open_input(&fmt, nullptr, nullptr, &opts);
        av_dict_free(&opts);
        if (e < 0) {
            fmt = nullptr;
            *why = "not a recognised audio format (" + averr(e) + ")";
            return RAD_E_INVAL;
        }
        e = avformat_find_stream_info(fmt, nullptr);
        if (e < 0) { *why = "could not read the stream headers (" + averr(e) + ")"; return RAD_E_INVAL; }
        const AVCodec* codec = nullptr;
        stream = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
        if (stream < 0 || !codec) { *why = "no sound in it: the media holds no audio stream"; return RAD_E_INVAL; }
        dec = avcodec_alloc_context3(codec);
        if (!dec) { *why = "out of memory"; return RAD_E_NOMEM; }
        e = avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar);
        if (e < 0) { *why = "decoder setup failed (" + averr(e) + ")"; return RAD_E_INVAL; }
        dec->thread_count = 1;
        e = avcodec_open2(dec, codec, nullptr);
        if (e < 0) { *why = std::string("no decoder for ") + codec->name + " (" + averr(e) + ")"; return RAD_E_UNSUPPORTED; }
        return RAD_OK;
    }
};

}  /* namespace */

int decode_audio(const uint8_t* data, size_t n, int32_t rate, int64_t max_samples,
                 std::vector<float>* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    out->clear();
    if (rate <= 0 || max_samples <= 0) { *why = "no sample rate or length to decode to"; return RAD_E_INVAL; }
    AudioIn in;
    RAD_TRY(in.open(data, n, why));
    AVCodecContext* dec = in.dec;
    const int channels = dec->ch_layout.nb_channels > 0 ? dec->ch_layout.nb_channels : 1;
    if (dec->sample_rate <= 0) { *why = "the audio stream states no sample rate"; return RAD_E_INVAL; }

    /* Resampled at the source's own layout, float interleaved; the mono mean is taken after. A
     * layout the stream does not state is taken as `channels` unordered channels. */
    AVChannelLayout layout{};
    if (av_channel_layout_check(&dec->ch_layout)) av_channel_layout_copy(&layout, &dec->ch_layout);
    else av_channel_layout_default(&layout, channels);
    int e = swr_alloc_set_opts2(&in.swr, &layout, AV_SAMPLE_FMT_FLT, rate, &layout,
                                (AVSampleFormat)dec->sample_fmt, dec->sample_rate, 0, nullptr);
    av_channel_layout_uninit(&layout);
    if (e < 0 || !in.swr || swr_init(in.swr) < 0) {
        *why = "no resampler for this audio (" + averr(e) + ")";
        return RAD_E_UNSUPPORTED;
    }

    std::vector<float> buf;
    auto take = [&](const uint8_t** src, int n_in) -> int {
        const int cap = swr_get_out_samples(in.swr, n_in);
        if (cap <= 0) return 0;
        buf.resize((size_t)cap * (size_t)channels);
        uint8_t* dst = reinterpret_cast<uint8_t*>(buf.data());
        const int got = swr_convert(in.swr, &dst, cap, src, n_in);
        if (got < 0) { *why = "resampling failed (" + averr(got) + ")"; return RAD_E_INVAL; }
        for (int i = 0; i < got && (int64_t)out->size() < max_samples; ++i) {
            float s = 0.0f;
            for (int c = 0; c < channels; ++c) s += buf[(size_t)i * channels + c];
            out->push_back(channels == 1 ? s : s / (float)channels);
        }
        return 0;
    };

    AVPacket* pkt = av_packet_alloc();
    AVFrame* f = av_frame_alloc();
    if (!pkt || !f) {
        av_packet_free(&pkt);
        av_frame_free(&f);
        *why = "out of memory";
        return RAD_E_NOMEM;
    }
    int st = RAD_OK;
    bool done = false;
    while (!done && (int64_t)out->size() < max_samples) {
        e = av_read_frame(in.fmt, pkt);
        if (e < 0) {
            avcodec_send_packet(dec, nullptr);   /* drain */
            done = true;
        } else {
            if (pkt->stream_index == in.stream) e = avcodec_send_packet(dec, pkt);
            av_packet_unref(pkt);
            if (e < 0 && e != AVERROR(EAGAIN)) { *why = "the audio would not decode (" + averr(e) + ")"; st = RAD_E_INVAL; break; }
        }
        while ((e = avcodec_receive_frame(dec, f)) >= 0) {
            if ((st = take(const_cast<const uint8_t**>(f->extended_data), f->nb_samples)) < 0) break;
            av_frame_unref(f);
        }
        if (st < 0) break;
        if (e != AVERROR(EAGAIN) && e != AVERROR_EOF) { *why = "the audio would not decode (" + averr(e) + ")"; st = RAD_E_INVAL; break; }
    }
    /* What the resampler still holds. */
    if (st == RAD_OK && (int64_t)out->size() < max_samples) st = take(nullptr, 0);
    av_packet_free(&pkt);
    av_frame_free(&f);
    if (st < 0) return st;
    if (out->empty()) { *why = "the audio decoded to no samples"; return RAD_E_INVAL; }
    return RAD_OK;
}

#endif  /* RAD_HAVE_FFMPEG */

}  /* namespace mm */
}  /* namespace rad */
