# Radiance in a container

The repository builds into one Docker image that holds the engine, its plugins, its tools and the
recipes, and nothing else: it is assembled from scratch out of the install tree and exactly the
shared libraries that tree loads. The host needs Docker, an AMD GPU and an `amdgpu` kernel driver
that supports ROCm 7.2. It does not need ROCm itself, a compiler, or ffmpeg.

The image is published as [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance),
with its kernels built for `gfx1201`: `docker pull stilldeadcode/radiance`. Build it yourself for
other cards or another commit. The examples below use `radiance`, the tag a local build gets; with
the published image, write `stilldeadcode/radiance` instead.

Besides radiance (Apache-2.0, `/opt/radiance/share/doc/radiance/`), the image holds glibc,
libstdc++, the ROCm runtime, FFmpeg, busybox and a few other libraries, each under its own license,
GPL and LGPL among them. `/usr/share/doc/THIRD-PARTY` in the image lists every one with its exact
version, its license files and where its source is; `docker/licenses.sh` writes it during the build
and stops the build if a file in the image cannot be traced to a package.

## Build the image

```sh
docker/build.sh                        # HEAD, tagged radiance:<version>, radiance:<commit>, radiance:latest
docker/build.sh -s gpu-host            # stream the commit over ssh; gpu-host builds and keeps the image
docker/build.sh -j 8                   # cap compile jobs (a device kernel takes 2-3 GB under hipcc)
docker/build.sh -t 'gfx1200;gfx1201'   # GPU architectures to compile kernels for (default gfx1201)
docker/build.sh -r <commit>            # any other commit
docker/build.sh --target build         # stop at a stage, tagged radiance-<stage>
```

The context the script sends is `git archive` of the commit, so the image holds exactly what is
committed: no build directory, no uncommitted edit, no model file. It builds with BuildKit
(`docker buildx`), which runs independent stages side by side and caches each one; with `-s` only
the remote machine needs buildx.

The `Dockerfile` has three stages:

| stage | what it does |
|---|---|
| `ffmpeg` | FFmpeg 8.1 with only the demuxers `core/mm/media.cpp` admits and the decoders their streams carry (still images, H.264, HEVC, VP8, VP9, AV1 through dav1d, MPEG-1/2/4, Theora, FLV/H.263). The distribution's build brings about 200 packages of encoders, filters and display libraries the engine never calls |
| `build` | `rocm/dev-ubuntu-24.04:7.2.4` plus cmake, ninja and g++-14. Configures with the kernels compiled for `GPU_TARGETS`, builds, and installs to `/opt/radiance`. Device code is not stored compressed: the engine's AQL backend reads the fat binaries itself and refuses a compressed bundle |
| `test` | `ctest -LE gpu`: every suite that runs without a card. The image is only produced if they pass |
| `rootfs` | The install tree, stripped, plus every shared library its binaries and plugins load (resolved by `ldd`, glibc and the ROCm runtime included) in one directory the loader searches by default, `amdgpu.ids`, and a static busybox. Every ELF file is then resolved from inside a chroot of the tree, so the build fails on a library the image lacks |
| `runtime` | `FROM scratch` and the rootfs. This is the image that ships |

The largest single file is comgr (about 150 MB), which the HIP runtime cannot initialise a device
without. There is no distribution in the shipped image: no package manager and no shell beyond busybox
(`docker exec -it <name> sh`). JPEG XL input is refused by name, since its decoder needs libjxl.

A build container has no card to enumerate, so the GPU architecture must be named. A bare-metal
`./build.sh` detects it instead. `libr4d` builds for gfx12 targets only; any other target in the
list just gets no RDNA4 kernels.

The ROCm version in the image has to be one the host's kernel driver supports. It is a build
argument (`ROCM_VERSION`, default 7.2.4).

The GPU suites (`mem_test`, `r4d_selftest`, `kbench`) need the card to themselves. They run from
the `build` stage:

```sh
docker/build.sh --target build
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    radiance-build ctest --test-dir /build -L gpu --output-on-failure
```

## Run it

The entrypoint is `radiance`. Every other tool (`rad-convert`, `rad-info`, `rad-kbench`,
`rad-tune`, `rad-schemas`, `rad-tokcheck`) is on `PATH`, and the recipes are in
`/opt/radiance/share/radiance/recipes/`.

```sh
docker run --rm -v /srv/models/rad:/models:ro --entrypoint rad-info radiance /models/m.rad

docker run -d --name radiance --network host --init \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v /srv/models/rad:/models:ro \
    radiance --model /models/m.rad --tp 2 --host :: --port 8000
```

Every one of these options is needed:

- **`--device /dev/kfd --device /dev/dri`**, plus the `render` and `video` groups, give the
  container the GPUs.
- **`--security-opt seccomp=unconfined`**: Docker's default seccomp profile refuses io_uring.
  Qwen3.8-Flash-Next's n-gram embedding table is read from the container through io_uring, so
  without this the model cannot serve. The profile also refuses the NUMA binding calls the ROCm
  runtime makes for host memory.
- **`--init`**: the engine is not PID 1. It finishes its step and releases the cards on SIGTERM.
- **`--ulimit memlock=-1`**: where no device kernel gathers the n-gram table (a run on the host
  backend), `--ngram-placement ram` locks a copy of it in RAM -- 47.7 GiB for
  Qwen3.8-Flash-Next, far past Docker's default memlock limit -- and without the limit raised
  `auto` reads it from the container and `ram` refuses the start. On the cards the copy is pinned
  memory the cards read, which the limit does not count. `deploy/compose/common.yaml` sets it.
- **`--network host`**: the server binds the host's addresses directly, IPv4 and IPv6, with no
  NAT in front of it.

The image sets `GPU_MAX_ALLOC_PERCENT=100` and `GPU_MAX_HEAP_SIZE=100`. The HIP runtime's defaults
cap one allocation, and the heap, below what the placement plan hands out.

**A plugin of your own** — a kernel library for another card, an architecture, a quantiser — is
mounted and put first on the search path; nothing in the image changes:

```sh
docker run ... -v /srv/radiance-plugins:/plugins:ro \
    -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    radiance --model /models/m.rad
```

`/plugins` holds any of `kernels/`, `architectures/` and `quantizers/`, built against the same
radiance version (`docs/PLUGIN.md` §13). The engine in the image starts on any AMD GPU family: the
image's `GPU_TARGETS` are what its bundled kernel library, libr4d, is compiled for, not a limit on
the cards the engine runs on.

## Compose

`deploy/compose/` has one file per model and a `common.yaml` that each of them extends:

| file | container | what it is |
|---|---|---|
| `flashnext.yaml` | `qwen3.8-next-flash-fp8-iq4r-moe.rad` | Qwen3.8-Flash-Next, 4-bit codebook experts and an int8 trunk; MTP depth 3; idle sessions kept in host memory and on disk |
| `flashnext-bf16.yaml` | `qwen3.8-flash-next-bf16.rad` | Qwen3.8-Flash-Next in bf16, streamed from the disk tier. A reference, not a production server |
| `qwen3.6-35b-a3b.yaml` | `qwen3.6-35b-a3b-fp8.rad` | Qwen3.6-35B-A3B, FP8 |
| `qwen3.8-27b-fp8.yaml` | `qwen3.8-27b-fp8.rad` | Qwen3.8-27B FP8 with its DFlash2 drafter and vision tower |
| `qwen3.8-27b-mxfp4.yaml` | `qwen3.8-27b-mxfp4.rad` | Qwen3.8-27B with AMD's Quark AWQ-MXFP4 trunk, its DFlash2 drafter and vision tower |
| `qwen3.8-27b-bf16.yaml` | `qwen3.8-27b-bf16.rad` | Qwen3.8-27B in bf16 |
| `minicpm5-2b.yaml` | `minicpm5-2b-fp8.rad` | MiniCPM5-2B FP8 with its DSpark drafter |
| `minicpm5-2b-bf16.yaml` | `minicpm5-2b-bf16.rad` | MiniCPM5-2B in bf16 with its DSpark drafter |

The container names are the files the README's model table downloads or converts.

Copy the directory to the serving host and fill in `.env` from `.env.example`. It sets the image,
the models directory (mounted read-only at `/models`), a writable state directory (`/data`, where
the prefix cache's disk tier lives), the API key, the port, and the user and GPU groups the server
runs as. The models directory has to be on a local filesystem that does direct I/O, such as ext4,
XFS or btrfs: Qwen3.8-Flash-Next reads its n-gram table from the container that way. On btrfs with
compression on, the extents it stored compressed are read through the page cache instead, which
works but loses what the direct reads are for; a container copied into a directory set `chattr +C`
is stored uncompressed. Then:

```sh
docker compose -f flashnext.yaml up -d      # start; restarts with the Docker daemon
docker compose -f flashnext.yaml logs -f    # follow the log
docker compose -f flashnext.yaml down       # stop
```

Each model takes both cards and the same port, so run one at a time: `down` one before `up` on
another. The server runs as the host user given in `.env`, so the files it writes under `/data`
belong to that user. `/health`, `/ping` and the dashboard page answer without the API key, and the
health check fetches `/health` with busybox's `wget`.
