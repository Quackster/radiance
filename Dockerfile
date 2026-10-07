# syntax=docker/dockerfile:1
#
# radiance as a container image: the engine, its plugins and its tools compiled inside the ROCm
# toolchain image, the host test suite run against that build, and a runtime image built FROM
# SCRATCH that holds the install tree and exactly the shared libraries it loads -- no distribution,
# no package manager, no shell beyond a static busybox.
#
#   docker/build.sh                                       # the usual way (BuildKit / buildx)
#   docker buildx build -t radiance .                     # the same, by hand
#   docker buildx build --build-arg GPU_TARGETS="gfx1200;gfx1201" -t radiance .
#
# Stages:
#   ffmpeg   FFmpeg built with only the demuxers core/mm/media.cpp admits and the decoders their
#            streams need (pictures, video, and audio with swresample); the distribution's build
#            brings ~200 packages of encoders, filters and display libraries the engine never calls
#   build    ROCm dev image + cmake/ninja/g++-14; configures, builds, installs to /stage
#   test     `ctest -LE gpu` on that build -- everything that does not need a card
#   rootfs   the install tree stripped, and every library its binaries and plugins load, resolved
#            by ldd, copied into one directory and checked from inside a chroot
#   runtime  FROM scratch + rootfs. This is the image that ships
#
# Device code is NOT built with --offload-compress: the engine's AQL backend reads the fat binaries
# itself and refuses a compressed bundle.
#
# GPU_TARGETS must be named. A bare-metal ./build.sh asks the machine which cards it has; a build
# container has no card to ask, and a build that guessed would emit kernels for hardware nobody
# chose. They are the targets the shipped KERNEL LIBRARIES are compiled for -- libr4d serves RDNA4
# only and drops every architecture outside gfx12 on its own. The engine itself runs on every AMD
# GPU family whatever is named (its own kernels carry the generic family targets), so a kernel
# library for another card is added to the image by mounting it on the search path:
#   docker run ... -v /my/plugins:/plugins -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance
#
# The ROCm userspace in the image must be one the host's amdgpu kernel driver supports. It is
# pinned to the version the serving host runs; move both together.

ARG ROCM_VERSION=7.2.4
ARG UBUNTU_VERSION=24.04
ARG FFMPEG_VERSION=8.1.3
ARG FFMPEG_SHA256=7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3

# ----------------------------------------------------------------------------------------- ffmpeg
FROM ubuntu:${UBUNTU_VERSION} AS ffmpeg
ARG FFMPEG_VERSION
ARG FFMPEG_SHA256
ARG JOBS=
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential nasm pkg-config ca-certificates curl xz-utils zlib1g-dev libdav1d-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp
# The demuxers are media.cpp's kFormats whitelist, the only containers it opens. The decoders are
# what those containers carry: still images, H.264 and HEVC (MP4, MOV, HEIF), VP8 and VP9 (WebM,
# WebP), AV1 through dav1d (WebM, MP4, AVIF), MPEG-1/2/4, Theora and the FLV/H.263 family. JPEG XL
# is recognised and refused by name: its only decoder is libjxl's. No protocols: media.cpp reads
# from memory through its own I/O context.
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz \
 && echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c - \
 && tar -xJf ffmpeg.tar.xz && cd "ffmpeg-${FFMPEG_VERSION}" \
 && ./configure --prefix=/opt/ffmpeg --enable-shared --disable-static --disable-debug \
      --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything \
      --disable-avdevice --disable-avfilter --enable-swresample \
      --enable-zlib --enable-libdav1d \
      --enable-demuxer=mov,matroska,avi,mpegts,flv,ogg,gif,apng,image2pipe,image_png_pipe,image_jpeg_pipe,image_jpegls_pipe,image_jpegxl_pipe,image_webp_pipe,image_bmp_pipe,image_tiff_pipe,image_qoi_pipe,image_pgm_pipe,image_ppm_pipe,image_pbm_pipe,image_pam_pipe,image_gif_pipe,wav,w64,mp3,flac,aac,aiff \
      --enable-decoder=h264,hevc,vp8,vp9,libdav1d,mpeg4,mpeg1video,mpeg2video,mjpeg,jpegls,png,apng,gif,webp,bmp,tiff,qoi,pgm,ppm,pbm,pam,pgmyuv,theora,vp3,flv,h263,pcm_u8,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_f32le,pcm_f32be,pcm_f64le,pcm_alaw,pcm_mulaw,mp3float,mp2float,flac,vorbis,opus,aac \
      --enable-parser=h264,hevc,vp8,vp9,av1,mpeg4video,mpegvideo,mjpeg,png,gif,bmp,webp,vp3,qoi,pnm,h263,jpegxl,mpegaudio,flac,aac,vorbis,opus \
 && make -j"${JOBS:-$(nproc)}" \
 && make install \
 && mkdir -p /opt/ffmpeg/share/doc/ffmpeg \
 && cp LICENSE.md COPYING.LGPLv2.1 /opt/ffmpeg/share/doc/ffmpeg/ \
 && sed -n 's/^#define FFMPEG_CONFIGURATION "\(.*\)"$/\1/p' config.h \
      > /opt/ffmpeg/share/doc/ffmpeg/CONFIGURATION \
 && echo "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz, sha256 ${FFMPEG_SHA256}" \
      > /opt/ffmpeg/share/doc/ffmpeg/SOURCE

# ------------------------------------------------------------------------------------------ build
FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION} AS build
ARG GPU_TARGETS=gfx1201
# Parallel compile jobs. Empty means one per core; a device-kernel translation unit takes 2-3 GB
# of RAM under hipcc, so a host that is also serving wants a lower number.
ARG JOBS=

# g++-14 rather than the image's default 13: its libstdc++ is the one hipcc then compiles the
# device-side translation units against too, and it is the runtime image's libstdc++ as well.
# gawk because tools/ops_doc_check.sh uses gensub, which Ubuntu's default awk (mawk) lacks.
# libdav1d7 is the runtime half of the AV1 decoder the ffmpeg stage linked; busybox-static goes
# into the runtime image.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake ninja-build pkg-config g++-14 gawk libdav1d7 busybox-static \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg /opt/ffmpeg /opt/ffmpeg
RUN echo /opt/ffmpeg/lib > /etc/ld.so.conf.d/ffmpeg.conf && ldconfig

ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm CC=gcc-14 CXX=g++-14 \
    PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig

WORKDIR /src
COPY . .

# The install prefix is fixed at configure time because the default $RADIANCE_HOME is compiled in
# as <prefix>/share/radiance; DESTDIR then stages the tree without moving that default.
# oom_score_adj 1000: on a host that is also serving, the compiler is what the OOM killer takes,
# not the server.
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    cmake -S . -B /build -G Ninja \
      -DCMAKE_INSTALL_PREFIX=/opt/radiance \
      -DRAD_WITH_HIP=ON -DRAD_WITH_FFMPEG=ON \
      "-DRAD_GPU_TARGETS=${GPU_TARGETS}" \
 && cmake --build /build ${JOBS:+-j${JOBS}} \
 && DESTDIR=/stage cmake --install /build \
 && mkdir -p /stage/opt/radiance/share/radiance/recipes \
 && cp data/recipes/*.recipe /stage/opt/radiance/share/radiance/recipes/

# ------------------------------------------------------------------------------------------- test
FROM build AS test
# Host-only: a build container has no /dev/kfd. The `gpu`-labelled suites (mem_test,
# r4d_selftest, kbench) run against a card from this stage:
#   docker buildx build --target build -t radiance-build . && docker run --rm \
#       --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined radiance-build \
#       ctest --test-dir /build -L gpu --output-on-failure
# One test at a time: several suites assert that a cost scales linearly, a ratio of two timings
# that tests running beside them would distort. The build sandbox forbids io_uring, so avx_check
# reports its file-backed gather case as skipped here. The `oot` suite -- every shipped plugin
# rebuilt against the installed package, libr4d included -- is the development tree's to run
# (tests/oot_plugins.sh): it re-does this stage's whole compile.
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    ctest --test-dir /build -LE 'gpu|oot' --output-on-failure \
 && touch /build/host-tests-passed

# ----------------------------------------------------------------------------------------- rootfs
FROM build AS rootfs
# The marker makes this stage, and so the shipped image, depend on the test stage.
COPY --from=test /build/host-tests-passed /rootfs/opt/radiance/share/radiance/.host-tests-passed
# Every library the binaries and plugins load, plus comgr, which the HIP runtime opens at run time
# and cannot initialise a device without, goes into one directory the dynamic loader searches by
# default: no ld.so.cache, no LD_LIBRARY_PATH. libdrm_amdgpu names the card from amdgpu.ids. A
# static busybox provides the health check's wget and a shell for `docker exec`. docker/licenses.sh
# traces each of those files to its package and records its license and source in
# /usr/share/doc/THIRD-PARTY. Then every ELF file is resolved from inside the tree, so nothing can
# depend on a library the image lacks.
RUN set -e; R=/rootfs; L=$R/usr/lib/x86_64-linux-gnu; \
    mkdir -p $L $R/lib64 $R/usr/bin $R/bin $R/usr/share/libdrm $R/data $R/tmp; \
    chmod 1777 $R/tmp; \
    cp -a /stage/opt/radiance $R/opt/; \
    elves=$(find $R/opt/radiance -type f -exec sh -c 'head -c4 "$1" | grep -q ELF' _ {} \; -print); \
    strip --strip-unneeded $elves; \
    libs=$(for f in $elves /opt/rocm/lib/libamd_comgr.so.3; do ldd "$f"; done \
           | awk '/=> \//{print $3} /^\t\/lib64\//{print $1}' | sort -u); \
    for lib in $libs; do \
      case "$lib" in */ld-linux-x86-64.so.2) cp -L "$lib" $R/lib64/ ;; *) cp -L "$lib" $L/ ;; esac; \
    done; \
    cp -L /opt/rocm/lib/libamd_comgr.so.3 $L/; \
    cp /usr/share/libdrm/amdgpu.ids $R/usr/share/libdrm/; \
    cp /bin/busybox $R/usr/bin/busybox; ln -s /usr/bin/busybox $R/bin/sh; \
    sh /src/docker/licenses.sh $R $libs /opt/rocm/lib/libamd_comgr.so.3 \
      /usr/share/libdrm/amdgpu.ids /bin/busybox; \
    for f in $(cd $R && find opt/radiance usr/lib -type f -exec sh -c 'head -c4 "$1" | grep -q ELF' _ {} \; -print); do \
      out=$(chroot $R /lib64/ld-linux-x86-64.so.2 --list "/$f" 2>&1) || { echo "$f: $out" >&2; exit 1; }; \
      if echo "$out" | grep -q 'not found'; then echo "$f: $out" >&2; exit 1; fi; \
    done; \
    chroot $R /opt/radiance/bin/radiance --help > /dev/null; \
    du -sh $R $L $R/opt/radiance

# ---------------------------------------------------------------------------------------- runtime
FROM scratch AS runtime
ARG GPU_TARGETS=gfx1201
ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.title="radiance" \
      org.opencontainers.image.description="LLM inference engine for AMD GPUs" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      radiance.gpu-targets="${GPU_TARGETS}"

COPY --from=rootfs /rootfs/ /

# A single allocation may take the whole card and the HIP heap may grow to all of VRAM; the
# runtime's defaults cap both below what the placement plan hands out.
ENV PATH=/opt/radiance/bin:/usr/bin:/bin \
    GPU_MAX_ALLOC_PERCENT=100 \
    GPU_MAX_HEAP_SIZE=100

WORKDIR /data
EXPOSE 8000
ENTRYPOINT ["radiance"]
CMD ["--help"]
