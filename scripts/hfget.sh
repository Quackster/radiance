#!/bin/sh
# hfget.sh -- fetch a Hugging Face repo with curl and jq, and nothing else.
#
# The engine and its tooling do not depend on Python, so `huggingface-cli` is not assumed. The API
# gives a file listing as JSON and every file has a stable resolve/ URL, which is all a downloader
# needs.
#
#   scripts/hfget.sh <repo-id> [dest-dir] [revision]
#
# Set HF_TOKEN for a gated or private repo. Re-running resumes: a file whose local size already
# matches the listing is skipped and a short one is continued with `curl -C -`, so an interrupted
# transfer of a 300 GB checkpoint costs only what it had not finished.
#
# A CHANGED FILE IS FETCHED AGAIN even at the same size. A complete run records each file's content
# key (the LFS sha256, or the size for a small file) in <dest>/.hfget-manifest, and a later run
# compares the listing against it. The manifest is written only when every file arrived, so its
# presence means the directory is a complete copy of some revision.
#
# Exit status: 0 complete; 3 nothing on disk was changed (the listing could not be read, or what is
# still to fetch does not fit), so a complete earlier copy is intact; anything else, a transfer
# failed and a file may be partial -- re-run to resume. `radiance --model <repo-id>` serves the
# earlier copy on 3 and refuses on the rest.
#
# WEIGHT DUPLICATES ARE SKIPPED. Many repos carry the same tensors several times over -- safetensors
# beside the older pickle .bin, an ONNX export, a GGUF conversion -- and on a large checkpoint each
# copy is hundreds of gigabytes of the same numbers. When the listing holds any .safetensors, every
# other weight format is left behind.
#
# HFGET_JOBS concurrent transfers, default 4. These are long sequential reads: a few streams
# saturate the link and more only makes the destination seek.
set -e

# ---- child mode: one file, named by "<oid><tab><size><tab><path>" in $1, everything else inherited
# sh cannot export a function, so parallelism is one re-entry of this script per file under
# `xargs -P` rather than a worker pool.
if [ -n "$HFGET_CHILD" ]; then
    oid=${1%%	*}; rest=${1#*	}; size=${rest%%	*}; path=${rest#*	}
    key=${oid:-$size}                                  # LFS: sha256, non-LFS: size
    out=$HFGET_DEST/$path
    man=$HFGET_DEST/.hfget-manifest
    rec=""; [ -f "$man" ] && rec=$(p="$path" awk -F'\t' '$2==ENVIRON["p"]{print $1; exit}' "$man")
    if [ -f "$out" ] && [ "$(wc -c < "$out")" = "$size" ]; then
        if [ -z "$rec" ] || [ "$rec" = "$key" ]; then  # warm: size ok AND content-key matches (or unrecorded)
            echo "  = $path"
            printf '%s\t%s\n' "$key" "$path" >> "$HFGET_MAN_OUT"
            exit 0
        fi
        rm -f "$out"                                   # SAME SIZE, NEW CONTENT: resume is size-based and useless -> fresh
        echo "  ~ $path (same size, new content)"
    elif [ -f "$out" ]; then
        echo "  + $path (resuming at $(wc -c < "$out") of $size)"
    else
        echo "  > $path"
    fi
    mkdir -p "$(dirname "$out")"
    if [ -n "$HF_TOKEN" ]; then
        curl -sSL --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
             -H "Authorization: Bearer $HF_TOKEN" \
             -o "$out" "https://huggingface.co/$HFGET_REPO/resolve/$HFGET_REV/$path"
    else
        curl -sSL --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
             -o "$out" "https://huggingface.co/$HFGET_REPO/resolve/$HFGET_REV/$path"
    fi
    got=$(wc -c < "$out")
    [ "$got" = "$size" ] || { echo "hfget: $path is $got bytes, listing says $size" >&2; exit 1; }
    printf '%s\t%s\n' "$key" "$path" >> "$HFGET_MAN_OUT"
    exit 0
fi

# ---- parent ------------------------------------------------------------------------------------
repo=$1
[ -n "$repo" ] || { echo "usage: hfget.sh <repo-id> [dest-dir] [revision]" >&2; exit 2; }
dest=${2:-$HOME/models/$repo}
rev=${3:-main}
jobs=${HFGET_JOBS:-4}
command -v jq >/dev/null || { echo "hfget: jq is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "hfget: curl is required" >&2; exit 2; }

echo "hfget: listing $repo@$rev"
# THE LISTING IS PAGINATED AND THE PAGE SIZE IS SMALL. A large checkpoint runs to a hundred-odd
# shards and the API returns fifty entries with a `Link: ...; rel="next"` header. Taking only the
# first page is the worst possible failure here: every file it did fetch is the right length, so
# the size check passes and the converter is handed a checkpoint missing two thirds of its
# tensors. Follow the cursor until there is no next link.
api="https://huggingface.co/api/models/$repo/tree/$rev?recursive=true&expand=true"
listing='[]'
page=0
while [ -n "$api" ]; do
    hdr=$(mktemp "${TMPDIR:-/tmp}/hfhdr.XXXXXX")
    # A failed request is reported here rather than by set -e, which would exit with curl's status
    # and say nothing; and with 3, since nothing has been written yet.
    st=0
    if [ -n "$HF_TOKEN" ]; then
        body=$(curl -sSL --fail-with-body -D "$hdr" -H "Authorization: Bearer $HF_TOKEN" "$api") \
            || st=$?
    else
        body=$(curl -sSL --fail-with-body -D "$hdr" "$api") || st=$?
    fi
    [ "$(printf '%s' "$body" | jq -r 'type' 2>/dev/null)" = "array" ] || {
        rm -f "$hdr"
        echo "hfget: listing $repo@$rev failed (curl exit $st)" >&2
        [ -n "$body" ] && {
            echo "hfget: $(printf '%s' "$body" | jq -r '.error // .message // .' 2>/dev/null)" >&2
            echo "hfget: a gated or private repo needs HF_TOKEN in the environment" >&2; }
        exit 3; }
    listing=$(printf '%s\n%s' "$listing" "$body" | jq -s 'add')
    api=$(tr -d '\r' < "$hdr" | sed -n 's/^[Ll]ink: *<\([^>]*\)>; *rel="next".*/\1/p' | head -1)
    rm -f "$hdr"
    page=$((page + 1))
    [ "$page" -gt 200 ] && { echo "hfget: listing did not terminate after 200 pages" >&2; exit 1; }
done
echo "hfget: $page page(s) of listing"
have_st=$(printf '%s' "$listing" | jq '[.[] | select(.path | endswith(".safetensors"))] | length')
sel='.[] | select(.type == "file")'
[ "$have_st" -gt 0 ] && sel="$sel"' | select((.path | test("\\.(bin|pth|msgpack|h5|onnx|onnx_data|gguf)$")) | not)'

list=$(mktemp "${TMPDIR:-/tmp}/hfget.XXXXXX")
trap 'rm -f "$list" "$dest/.hfget-manifest.tmp"' EXIT INT TERM
printf '%s' "$listing" | jq -r "$sel"' | "\(.lfs.oid // "")\t\(.size)\t\(.path)"' > "$list"

n=$(wc -l < "$list")
[ "$n" -gt 0 ] || { echo "hfget: the listing has no files" >&2; exit 3; }
need=$(awk -F'\t' '{s += $2} END {printf "%.1f", s / 1073741824}' "$list")
mkdir -p "$dest"
: > "$dest/.hfget-manifest.tmp"
# WHAT IS STILL TO FETCH, not the repo's size: a warm cache of a model larger than the free space
# left beside it must not read as a download that does not fit. A file is counted from its local
# length, and whole when the manifest says its content changed.
todo=0
# Split by expansion, as the child does: a tab is IFS whitespace, so `read` would merge the empty
# key of a non-LFS file into the next field.
while IFS= read -r line; do
    oid=${line%%	*}; rest=${line#*	}; size=${rest%%	*}; path=${rest#*	}
    cur=0
    if [ -f "$dest/$path" ]; then cur=$(wc -c < "$dest/$path"); fi
    if [ "$cur" = "$size" ] && [ -f "$dest/.hfget-manifest" ]; then
        rec=$(p="$path" awk -F'\t' '$2==ENVIRON["p"]{print $1; exit}' "$dest/.hfget-manifest")
        if [ -n "$rec" ] && [ "$rec" != "${oid:-$size}" ]; then cur=0; fi
    fi
    if [ "$cur" -lt "$size" ]; then todo=$((todo + size - cur)); fi
done < "$list"
free=$(df -BG --output=avail "$dest" | tail -1 | tr -dc '0-9')
echo "hfget: $n file(s), $need GiB -> $dest"
[ "$have_st" -gt 0 ] && echo "hfget: safetensors present, pickle weight formats skipped"
echo "hfget: $(awk -v t="$todo" 'BEGIN { printf "%.1f", t / 1073741824 }') GiB to fetch," \
     "$free GiB free on the destination filesystem"
awk -v t="$todo" -v f="$free" 'BEGIN { exit !(t / 1073741824 > f) }' \
    && { echo "hfget: THE DOWNLOAD DOES NOT FIT" >&2; exit 3; }

export HFGET_CHILD=1 HFGET_DEST="$dest" HFGET_REPO="$repo" HFGET_REV="$rev" HFGET_MAN_OUT="$dest/.hfget-manifest.tmp"
self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
tr '\n' '\0' < "$list" | xargs -0 -P "$jobs" -I{} sh "$self" "{}" \
    || { echo "hfget: incomplete -- re-run to resume" >&2; exit 1; }
mv "$HFGET_MAN_OUT" "$dest/.hfget-manifest"
echo "hfget: complete -- $(du -sh "$dest" | cut -f1) in $dest"
