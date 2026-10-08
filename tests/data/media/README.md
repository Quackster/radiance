# tests/data/media

One solid red picture in every still format `core/mm/media.cpp` admits, and one short solid red
clip in each container and codec it admits; `mm_test` decodes every one and checks it is red at
its size. They were made with FFmpeg 8.1 (libx264, libx265, libvpx, libsvtav1, libaom, libtheora,
libwebp):

```sh
I="-f lavfi -i color=c=red:s=32x32:d=1:r=1"
V="-f lavfi -i color=c=red:s=64x64:d=1:r=4"
ffmpeg $I -frames:v 1 red.png
ffmpeg $I -frames:v 2 -f apng red.apng
ffmpeg $I -frames:v 1 red.jpg
ffmpeg -f lavfi -i color=c=red:s=1788x282:d=1:r=1 -frames:v 1 red-1788x282.jpg
ffmpeg $I -frames:v 1 -c:v libwebp red.webp
ffmpeg $I -frames:v 1 red.gif
ffmpeg $I -frames:v 1 red.bmp
ffmpeg $I -frames:v 1 red.tiff
ffmpeg $I -frames:v 1 -c:v qoi red.qoi
ffmpeg $I -frames:v 1 -pix_fmt rgb24 red.ppm
ffmpeg $I -frames:v 1 -c:v jpegls red.jls
ffmpeg $V -frames:v 1 -c:v libaom-av1 -cpu-used 8 -f avif red.avif
ffmpeg $V -c:v libx264 -pix_fmt yuv420p red-h264.mp4
ffmpeg $V -c:v libx265 -pix_fmt yuv420p -tag:v hvc1 red-hevc.mp4
ffmpeg $V -c:v libsvtav1 -pix_fmt yuv420p red-av1.mp4
ffmpeg $V -c:v libvpx red-vp8.webm
ffmpeg $V -c:v libvpx-vp9 red-vp9.webm
ffmpeg $V -c:v libx264 -pix_fmt yuv420p red-h264.mkv
ffmpeg $V -c:v libx264 -pix_fmt yuv420p -f mpegts red-h264.ts
ffmpeg $V -c:v mpeg4 red-mpeg4.avi
ffmpeg $V -c:v libtheora red-theora.ogv
```

`red-1788x282.jpg` is 4:2:0 at a width the scaler's vectors do not divide: converted to RGB it
is written 12 bytes past width x 3 (`a_picture_of_an_awkward_width_converts_inside_its_buffer`).

The Theora clip repeats its first frame with empty packets, which is how Ogg Theora encodes a frame
that does not change.
