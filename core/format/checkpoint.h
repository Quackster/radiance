/* checkpoint.h -- a safetensors or GGUF checkpoint as a source of logical weights.
 *
 * A checkpoint is read twice in this engine: by rad-convert, which writes a container from it, and
 * by the loader, which serves a checkpoint whose encoding is trivial directly (spec §4.3). Both
 * need the same three things, and this is the one copy of each:
 *
 *   - the tensors, as one row-major namespace however many shards and drafters they came in;
 *   - the model's metadata, from config.json (and its siblings) or the GGUF header;
 *   - a LOGICAL WEIGHT for a declared name: the tensors the architecture plugin's name map
 *     resolves it to -- concatenations, alternatives and expert slices included -- with the
 *     encoding they already are when that encoding is trivial, and the rows of it on demand.
 *
 * Trivial is the set spec §4.3 names: plain f32, f16 and bf16 (GGUF's too), block-scaled FP8,
 * where `X.weight_scale_inv` is the scale plane of `X.weight`, and MXFP4 as AMD Quark ships it
 * (`X.weight` U8 codes, `X.weight_scale` U8 exponents), which is `fp4_e2m1*e8m0[1x32]` byte for
 * byte. Anything else is a weight whose served form needs quantisation work, which a recipe does
 * ahead of time -- or, for a published GPTQ-packed release (AutoRound, GPTQ), an IMPORT: the
 * shipped integer codes and group scales re-laid as an encoding the kernels read, at convert time.
 *
 * AN IMPORTED QUANTISATION IS STATED WEIGHT BY WEIGHT. The checkpoint's quantisation descriptor
 * (`auto-round`, `gptq`, `quark`) names the tool that made it, not a served form, so no
 * architecture plugin claims it; the plugin is chosen as Checkpoint::imports() describes.
 */
#pragma once
#include "../rad_internal.h"
#include "../rad_core.h"
#include "safetensors.h"
#include "../text/gguf.h"

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

class Program;

/* ================================================================== tensors */
/* One shape convention from here on: ROW-MAJOR, shape[0] slowest. GGUF stores ne[] fastest-first,
 * so it is reversed on the way in. Getting this backwards transposes every matrix in the model
 * and produces fluent wrong output, so it is done exactly once, in Checkpoint::open. */
struct SrcTensor {
    std::string          name;
    uint32_t             ggml = 0;
    uint32_t             rad_dtype = RAD_DT_INVALID;   /* INVALID: a GGUF block type */
    std::vector<int64_t> shape;                        /* row-major */
    const void*          data = nullptr;
    int64_t              bytes = 0;
    /* GPTQ-PACKED (auto_gptq's layout, which AutoRound's `auto_round:auto_gptq` export shares):
     * `data` is the qweight's int32 words [K * bits / 32, N], each packing 32 / bits codes along
     * K from the low bits up, stored unsigned with the zero point 2^(bits-1). The tensor this
     * describes is the logical [N, K] code matrix, `rad_dtype` RAD_I4 or RAD_I8; `gptq_group` is
     * the K columns one row of its scale tensor covers. 0 for any other tensor. */
    int32_t              gptq_bits = 0;
    int64_t              gptq_group = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return shape.empty() ? 0 : n;
    }
};

/* ================================================================== metadata */
/* The model's metadata as RadModelMeta, with the strings it points at owned here. */
struct MetaOwn {
    std::string arch, name, quant;
    std::vector<std::string> keys, vals;
    std::vector<const char*> kp, vp;
    RadModelMeta m{};

    void kv(const std::string& k, const std::string& v) { keys.push_back(k); vals.push_back(v); }
    /* A key given LAST that must also WIN: a lookup takes the first row with the key, so
     * appending a key the checkpoint already set would write both and read the old one. */
    void set(const std::string& k, const std::string& v);
    bool has(const std::string& k) const;
    const char* get(const std::string& k) const;     /* null when absent */
    /* Republish the pointers RadModelMeta carries. Every kv()/set() may move the strings, so this
     * runs after the last of them and before `m` is read. */
    void finish();
};

int meta_from_config_json(const std::string& dir, MetaOwn* out);
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out);
int meta_merge_generation_config(const std::string& dir, MetaOwn* out);
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out);
int meta_from_gguf(GgufFile& g, MetaOwn* out);

/* ================================================================== the checkpoint */
class Checkpoint {
public:
    Checkpoint() = default;
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    /* A .gguf, a .safetensors, a model.safetensors.index.json, or the directory holding one --
     * and the metadata beside it. */
    int open(const std::string& path);
    /* A SECOND checkpoint, merged into this one's namespace and metadata under `prefix`: a
     * drafter that ships as its own repository (DFlash2) borrows the target's embedding and head,
     * so it belongs in the target's container, its tensors under `draft.` and its config.json
     * under `draft.` keys. */
    int add_draft(const std::string& dir, const std::string& prefix);

    const SrcTensor* find(const std::string& n) const;
    const std::vector<SrcTensor>& tensors() const { return tensors_; }
    bool               is_gguf() const { return kind_ == Gguf; }
    GgufFile&          gguf() { return gguf_; }
    const std::string& path() const { return path_; }
    const std::string& dir() const { return dir_; }   /* where config.json and tokenizer.json sit */
    MetaOwn&           meta() { return meta_; }
    const MetaOwn&     meta() const { return meta_; }

    /* Sub-tensor `index` of a STACKED tensor -- dim 0 of a rank >= 2 tensor -- as a view, owned
     * here so every pointer handed out stays valid for the checkpoint's life. Null, with the
     * reason logged, when there is no such slice. */
    const SrcTensor* slice(const SrcTensor& st, int index, const char* declared);

    /* FP8's block, from quantization_config.weight_block_size; 128x128 when it says nothing. */
    int64_t fp8_block_rows() const { return fp8_br_; }
    int64_t fp8_block_cols() const { return fp8_bc_; }

    /* THE CHECKPOINT'S QUANTISATION, IMPORTED WEIGHT BY WEIGHT, or None. Read from
     * quantization_config at open: AutoRound's `auto_round:auto_gptq` export and GPTQ's own (v1,
     * the zero point stored less one) are Gptq; Quark's MXFP4 (fp4 weights, per_group 32, e8m0
     * scales, no per-layer overrides) is QuarkMx. A quantisation of one of those tools in a form
     * this does not read stays None, with the reason in import_refusal(). */
    /* Its quantization_config keys are then renamed `imported_quantization_config.*`: they
     * describe the tool, not a served format, and a plugin reads the latter there. */
    enum class Import { None, Gptq, QuarkMx };
    Import imports() const { return import_; }
    const std::string& import_refusal() const { return import_why_; }

    /* The pieces an imported weight is read from, as views owned here: for `name` (a `.weight`)
     * the GPTQ-packed codes and their scale tensor, or Quark's codes and exponents as their RAD
     * dtypes. RAD_E_NOTFOUND when the checkpoint ships `name` as an ordinary tensor or not at
     * all; another error, with `why`, when it ships an imported form this cannot read. */
    int import_piece(const std::string& name, const SrcTensor** codes, const SrcTensor** scale,
                     std::string* why);

private:
    int add_shard(const std::string& p, const std::string& prefix = "");
    int open_safetensors(const std::string& path);
    int open_gguf(const std::string& path);
    int alias_mxfp4();
    void detect_import();

    enum Kind { None, Gguf, SafeT } kind_ = None;
    std::string path_, dir_;
    GgufFile    gguf_;
    MetaOwn     meta_;
    std::vector<std::unique_ptr<SafeTensorsFile>> shards_;
    std::vector<SrcTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::deque<SrcTensor> sliced_;            /* views; a deque never moves what it holds */
    int64_t fp8_br_ = 128, fp8_bc_ = 128;
    Import      import_ = Import::None;
    std::string import_why_;
    /* import_piece's answers by name, so a weight resolved twice reads one pair of views. */
    std::unordered_map<std::string, std::pair<const SrcTensor*, const SrcTensor*>> imported_;
};

/* ================================================================== logical weights */
struct ParoLinear;                            /* a paroquant_mxfp6 source linear, defined below */

/* A declared weight's source, as the name map resolves it: pieces joined along `concat_dim`, and
 * for block-scaled FP8 each piece's scale plane beside it. */
struct CkptWeight {
    std::string name;                         /* the logical (declared) name */
    std::vector<const SrcTensor*> pieces;     /* the codes, in concatenation order */
    std::vector<const SrcTensor*> scales;     /* FP8 only: one a piece */
    int      concat_dim = 0;
    uint32_t rank = 0;
    int64_t  shape[RAD_MAX_RANK] = {0};       /* the logical shape */
    int64_t  rows = 0, cols = 0;              /* its [rows, cols] view (rad_enc_view) */
    /* PAROQUANT: one a piece, when that piece is a rotated linear of a `paroquant_mxfp6`
     * checkpoint; empty otherwise. Its shape is the LOGICAL [N, K], and ckpt_rows_f32 de-rotates
     * each piece through ckpt_paro_rows_f32 rather than reading the packed codes. */
    std::vector<std::shared_ptr<ParoLinear>> paro;
    /* The encoding the pieces ARE, when it is trivial; `why` says what is not when it is not. */
    bool        trivial = false;
    /* GPTQ-PACKED PIECES, read into `enc` (`i4` or `i8` codes, bf16 scales a 128 of K) by
     * ckpt_plane_rows: the codes transposed and offset to signed, the f16 scales rounded to bf16.
     * Not trivial -- the rounding means the numbers are not served as shipped -- so the loader
     * refuses one and rad-convert keeps it in `enc`, as it keeps a trivial weight. */
    bool        imported = false;
    RadEncoding enc{};
    std::string why;
};

/* Resolve `name` through `prog`'s name map. RAD_OK; RAD_E_NOTFOUND when the name map has no entry
 * for it yet (declare may add one later) or, for an optional weight, none of its tensors is
 * there; anything else with the reason in `why`. */
int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why);

/* The checkpoint tensor a piece or scale reads: its name without the "[index]" a slice of a
 * stacked tensor carries, so an expert reads the stack it was cut from. */
std::string ckpt_tensor_of(const SrcTensor& t);

/* Whether `w` reads any tensor named in `names` (checkpoint names, as ckpt_tensor_of gives them),
 * through a piece or a scale. Each name it reads is added to `hit` when that is given. */
bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit = nullptr);

/* Logical rows [r0, r0 + nr) of `w` as f32 [nr, cols], whatever the pieces are stored as: a
 * plain float widened, FP8 times its block scale, a GGUF block type dequantised. */
int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why);

/* Rows [r0, r0 + nr) of plane `k` of a TRIVIAL or IMPORTED `w`, in the plane's own elements, dense
 * into `dst`: the bytes a container entry holds for it. */
int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why);

/* ================================================================== paroquant_mxfp6
 *
 * THE PAROQUANT CHECKPOINT'S QUANTIZED LINEAR, read into the core's own terms. A quantised linear
 * ships five tensors: `<base>.weight` (U8, packed 6-bit E2M3 codes four to three bytes),
 * `.weight_scale` (U8 E8M0, one shared exponent a 32-wide MX group) and the rotation
 * `.pairs` (I16 [S,K]), `.theta` (F16 [S,K/2]) and `.channel_scales` (F16 [1,K]). The published
 * transform is w = R(v) . D; the core's givens encoding wants column-global pairs and the scale
 * on the left of the decoder's divide (enc_givens_paro_tables), so the reader globalizes the
 * pairs and stores 1/channel_scales. What it produces is an ordinary `affine` encoding with a
 * `givens` transform, decodable by enc_decode_rows like any container weight.
 *
 * The plane buffers are owned here; `data(k)` is enc plane k's canonical row-major bytes. The
 * codes and the E8M0 scale are NOT copied -- they point straight into the checkpoint's mapping,
 * since a convert holds one of these a linear and the codes are the whole weight. Build an
 * EncodedView over them while this lives. */
struct ParoLinear {
    std::string name;
    int64_t     rows = 0, cols = 0;
    int64_t     stages = 0;                         /* S, the pairs' first extent */
    int64_t     group = 0;                          /* the rotation group width */
    RadEncoding enc{};
    const void* code = nullptr;                     /* enc plane 0: packed E2M3 codes */
    const void* wscale = nullptr;                   /* enc plane 1: E8M0 exponents */
    std::vector<uint8_t> pairs;                     /* enc plane 2: i32 [S, K] global pairs */
    std::vector<uint8_t> angle;                     /* enc plane 3: f32 [S, K/2] angles */
    std::vector<uint8_t> tscale;                    /* enc plane 4: f32 [1, K], 1/channel_scales */

    const void* data(int k) const {
        switch (k) {
            case 0:  return code;
            case 1:  return wscale;
            case 2:  return pairs.empty()  ? nullptr : pairs.data();
            case 3:  return angle.empty()  ? nullptr : angle.data();
            case 4:  return tscale.empty() ? nullptr : tscale.data();
            default: return nullptr;
        }
    }
};

/* Read the paroquant_mxfp6 linear whose tensors are `<base>.weight` and its four companions.
 * RAD_OK; RAD_E_NOTFOUND when the base or a companion is absent; RAD_E_FORMAT/RAD_E_SHAPE when
 * they disagree or the rotation does not fit. `out->name` is `base`. */
int ckpt_paro_read(Checkpoint& ck, const std::string& base, ParoLinear* out, std::string* why);

/* Logical rows [r0, r0 + nr) of a read linear as f32 [nr, cols], through the core's givens
 * decoder. */
int ckpt_paro_rows_f32(const ParoLinear& w, int64_t r0, int64_t nr, float* out, std::string* why);

}  /* namespace rad */
