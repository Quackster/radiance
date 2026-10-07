# System One decisions

`POST /v1/systemone` supports the decision profile of LiquidAI d1. It reads a state, optional images, and
named questions, evaluates the final prompt position for each question, and returns typed
answers with zero generated tokens. Ordinary chat models and unknown decision profiles receive
HTTP 501.

## Convert and serve d1

Start with the original [d1 safetensors checkpoint](https://huggingface.co/LiquidAI/d1-3B). The explicit
`decision.type=d1` metadata selects the model-specific prompt and readout. In particular, the
original Hugging Face `config.json` does not itself declare this serving capability.

```sh
build/bin/rad-convert /path/to/d1-3B -o d1-3b-bf16.rad --set decision.type=d1

build/bin/radiance --model d1-3b-bf16.rad --port 8000
```

Conversion uses the LFM2 architecture plugin. The original multimodal safetensors checkpoint
includes the SigLIP2 image encoder and projector; conversion includes them automatically, with
no separate `mmproj` file. Existing text-only containers must be converted again to accept images.
Image decoding requires FFmpeg support; configure the build with `-DRAD_WITH_FFMPEG=ON` to require it.
LFM2 GGUF tensor mapping is not supported in this version, including BF16 GGUF. Use the original
safetensors checkpoint even when a GGUF copy is already available. BF16 is the baseline for checking
probability parity; quantized variants need their own numerical validation.
The LFM2 plugin supports one rank only (`--tp 1`, the default); tensor parallelism is not supported.

```sh
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice; please refund the duplicate charge.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Charges and refunds", "technical": "App faults"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Can wait", "Today", "Blocking now"]
    }
  }
}'
```

## Request and response

`state` is required and may be a string, any JSON value, or null. Non-string states are rendered
as indented JSON in their supplied key order. `questions` is a non-empty object whose values
have `type`, string `instructions`, and the criteria described below. Question names and choice
criteria retain insertion order. An optional `model` must name the loaded model.

`images` is an optional array of base64 data-URL strings. Images appear in array order before
the state; use `state: null` when the pictures are the entire state. The model must have a loaded
and enabled image encoder. A text-only container returns HTTP 501 for a non-empty image array.
Remote URLs and filesystem paths are not accepted.

```json
{
  "state": null,
  "images": ["data:image/png;base64,..."],
  "questions": {
    "animal": {
      "type": "choice",
      "instructions": "Which animal is visible?",
      "criteria": {"cat": "A cat", "dog": "A dog", "other": "Neither"}
    }
  }
}
```

The media processor applies the checkpoint's resize, patch, and image-token expansion rules.
Each image is prepared once and its encoded features are shared by the questions in that request.
Image tokens use ordinary sequential positions. A single image expands to
`<|image_start|>`, its `<image>` feature tokens, and `<|image_end|>`. Tiled images place a
`<|img_row_R_col_C|>` marker before each tile in row order, and `<|img_thumbnail|>` before the
thumbnail. These markers and all feature tokens count toward the prompt length; callers only
supply the image data URLs and do not insert the markers themselves.

JSON numbers use the server's numeric representation: very large integers may be rounded, and
float formatting can differ from Python. Pass a preformatted state string when those details
must match the Python prompt exactly.

| Type | Criteria | Answer fields |
|---|---|---|
| `noul` | Optional object with `true` and `false` descriptions | `type`, `noul` (probability of yes) |
| `choice` | Non-empty object mapping option names to descriptions | `type`, `choice`, `confidence`, `probabilities` |
| `score` | Array of 2–10 descriptions, lowest level first | `type`, `score`, `confidence`, `probabilities`, `legend` |

Choice descriptions can be strings or null; an empty or null description falls back to the option
name with underscores replaced by spaces. Noul descriptions can be strings or null. Score level
descriptions must be strings. The response is `{"answers": {name: answer}, "usage":
{"input_tokens": n, "output_tokens": 0}}`.

The readout follows the [published d1 reference](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py):
each answer scores its strongest single-token spelling, then a softmax over those answer scores
produces the distribution. Noul considers `yes`/`Yes`/`YES` and `no`/`No`/`NO`. Choices use the
reference's single-token alias allocation, including a leading-space alias when that is one token.
Scores read digit tokens and return the expected level index, which can lie between levels.
`confidence` is the maximum option probability, matching d1's Python API; it is not TypeSafe's
uniform-adjusted confidence. No additional temperature calibration is applied.

Questions retain independent causal state. When their common state and image prefix spans at least
256 tokens (or one prefill batch if smaller) and the combined prompts exceed one batch, the
scheduler evaluates the common prefix once, then branches using shared attention pages and
independent recurrent-state snapshots.
Short requests use ordinary batching. Reuse falls back to independent prefill when the model's
state geometry or available capacity cannot support it.
`input_tokens` counts the full submitted prompt for every question, including repeated state
and expanded image tokens. The maximum question count is bounded by the server admission queue, and every individual
prompt must fit the model context. Choices are bounded by the 1,028-entry reference fallback
codebook and by the tokenizer's supply of distinct single-token aliases.

Streaming, sampler parameters, and unrecognized fields are rejected. A request cancelled by a disconnected client
releases its scheduler requests through the same lifecycle as ordinary generation. Missing or
non-finite score results return an error rather than a distribution.

## Limits

d1 declares a 32,768-token context in `text_config.max_position_embeddings` in its
[official checkpoint config](https://huggingface.co/LiquidAI/d1-3B/blob/da1fe36a861f24690f27f622dca1d8688503d113/config.json#L81).
The [model card](https://huggingface.co/LiquidAI/d1-3B/blob/da1fe36a861f24690f27f622dca1d8688503d113/README.md#L81)
lists the same context length; neither source establishes a training sequence length or a
quality guarantee at that limit. Radiance counts the state, question, template, and expanded
images against this per-question bound. `--max-model-len` sets the serving limit; its default
uses the checkpoint's declared context. Oversized prompts receive HTTP 400 without truncation.
Image preparation stops as soon as accumulated image expansions exceed that limit.

The decoder accepts source images up to 67,108,864 pixels by default (`--max-source-pixels`).
Following the d1 runner, images above 1,048,576 pixels are downscaled to that area before
processing. Each 512-by-512 tile produces 256 feature tokens. With the checkpoint's maximum
of ten tiles and a thumbnail of at most 256 feature tokens, one image uses at most 2,816 feature
tokens plus 13 markers: 2,829 prompt tokens. Extreme aspect ratios that require a tile or
thumbnail above the 1,024-patch encoder segment limit receive HTTP 400 before GPU execution.

The default HTTP body limit is 512 MiB (`--max-body-mib`), including base64 image data and JSON.
BF16 results can vary slightly with batching and numerical kernels; probability parity is not
bit-for-bit equality with the Python reference.

## Execution

The LFM2 plugin normalizes only the final hidden rows needed by the request and projects only the
candidate vocabulary entries used by System One. Candidate logits retain the original BF16
rounding before conversion to response probabilities. Ordinary generation still uses the full
vocabulary head, including when generation and decision requests share a batch.

On GPU backends, encoded image features remain in a bounded device pool while their questions
execute. Pool overflow uses host staging. The pool and candidate-scoring buffers are included in
the engine's memory planning; the model container does not need reconversion for these optimizations.
The runtime ABI is version 16, so the engine and all plugins must be rebuilt together.
