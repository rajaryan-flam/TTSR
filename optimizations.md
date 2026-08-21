# `test_video.py` inference optimizations

This document explains every performance change made to `test_video.py` since the
original script, why each one was made, and how it was verified. Baseline for all
numbers below: the `ananya.mp4` clip (453 frames, downscaled to 384px), `TTSR.pt`,
on an RTX PRO 6000 Blackwell (96GB).

Original (unoptimized) baseline: **3.23 FPS model inference, 2.84 FPS end-to-end.**

## Background: why there was room to optimize

`test_video.py` calls `model(lr=lr_t, lrsr=lr_sr_t, ref=ref_t, refsr=ref_sr_t)` once
per video frame. That call goes through `TTSR.forward()` (`model/TTSR.py`), which
internally does, every single time it's called:

```python
_, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)   # depends on the CURRENT frame
_, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)  # depends only on the ref image
ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)  # depends only on the ref image
S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)
```

`test_video.py` loads **one** HR reference image before the loop and passes the same
`ref_t`/`ref_sr_t` tensors to `model(...)` on every one of the 453 frames. That means
everything derived purely from `ref`/`refsr` — two of the three `LTE` calls above, plus
several `SearchTransfer` internals — was being recomputed from scratch on every frame
even though the inputs never changed. That redundant work is the first thing this pass
removes. It's a correctness-preserving refactor, not an approximation: `model.LTE`,
`model.SearchTransfer`, and `model.MainNet` are plain stateless submodules with no
special handling in `TTSR.forward()`, so calling them directly and reusing cached
results is mathematically identical to the original call.

## 1. Cache the reference image's LTE features once (bit-exact)

**Change:** compute `ref_lv1, ref_lv2, ref_lv3` and `refsr_lv3` once, before the frame
loop, by calling `model.LTE(...)` directly. Inside the loop, only `lrsr_lv3` (which
genuinely changes every frame) is recomputed, then fed into
`model.SearchTransfer(...)` and `model.MainNet(...)` directly instead of going through
`model.forward()`.

**Why:** the ref image's `LTE` (VGG19) features are pure functions of `ref`/`refsr`,
which never change across the video. Recomputing them 453 times was wasted work.

**Verified bit-exact:** compared `model.forward(...)` against the decomposed
`model.LTE`/`model.SearchTransfer`/`model.MainNet` calls on identical random input
tensors — `torch.equal(sr_old, sr_new)` returned `True`, max abs diff `0.0`.

**Measured effect:** model inference 140.1s → 131.0s for 453 frames (3.23 → 3.46
FPS). Smaller than expected — see the profiling note below for why.

## 2. Cache `SearchTransfer`'s reference-side unfold/normalize too (bit-exact)

**Change:** `model/SearchTransfer.py`'s `forward()` also does several operations that
depend *only* on the (now-cached) ref features, every time it's called:

```python
refsr_lv3_unfold = F.normalize(F.unfold(refsr_lv3, kernel_size=(3,3), padding=1).permute(0,2,1), dim=2)
ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3,3), padding=1)
ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6,6), padding=2, stride=2)
ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12,12), padding=4, stride=4)
```

Only the `lrsr`-side unfold actually depends on the current frame. These four lines
were pulled out of the per-frame call and computed once, alongside step 1. The
`bis` (batch index select) helper is reused directly from `model.SearchTransfer.bis`
rather than duplicated, since it's a pure tensor operation with no module state.

**Why:** same reasoning as step 1 — `model/SearchTransfer.py` was not modified (it's
stateless with no learnable parameters, so replicating its logic in `test_video.py`
carries no risk), only the call site in `test_video.py` was restructured.

**Verified bit-exact:** same method as step 1, re-run against the fully decomposed
pipeline — `torch.equal` `True`.

**Measured effect:** model inference 131.0s → 129.2s (3.46 → 3.51 FPS). Small, because
these unfold/normalize ops are cheap relative to the per-frame work that's left (see
below).

### Why steps 1+2 only bought ~8%, not more

Profiling the decomposed per-frame pipeline (realistic shapes: `lr` 384×384, `lrsr`
1536×1536) shows where the remaining time actually goes:

| Stage | Time/frame | Share |
|---|---|---|
| `LTE(lrsr)` (must run every frame) | 0.013s | 4.3% |
| `SearchTransfer` per-frame part (dense correlation) | 0.119s | 40.0% |
| `MainNet` (upsampling network) | 0.166s | 55.7% |

The redundant ref-side `LTE`/unfold work removed in steps 1-2 was only ever a small
slice of total time. The two dominant costs — the dense correlation in
`SearchTransfer` and the `MainNet` upsampling network — both **must** run once per
frame regardless, since they depend on the current frame's content. There's no way to
cache them away; further speedups have to come from making those specific operations
faster (see steps 5-6).

## 3. `torch.backends.cudnn.benchmark = True`

**Why:** every frame is resized to the same `(lr_h, lr_w)` before hitting the network
(`test_video.py`'s preprocessing forces this), so every conv call in the network sees
a fixed input shape for the entire run. cuDNN can autotune once (small one-time cost
on the first frame) and reuse the fastest algorithm for that shape on every subsequent
frame, instead of picking a safe default every time.

**Risk:** technically, different cuDNN algorithms can differ by ~1 ULP from each
other (different floating-point summation order). In practice this is far below the
1/255 quantization step when the output is rounded to `uint8`, so the saved video is
unaffected. This is a different, much weaker caveat than "changes the output" — it's
listed for completeness, not because it was observed to matter.

## 4. `torch.no_grad()` → `torch.inference_mode()`

**Why:** `inference_mode()` is a strict superset of `no_grad()`'s guarantees (it also
skips version-counter bookkeeping used for in-place-mutation checks). The whole script
is pure inference — `model.eval()` is set, nothing is ever backpropagated, and no
tensor produced inside the block is used outside it in a way that needs autograd. Safe
drop-in with no downside.

> **Correction:** current `test_video.py` uses `torch.no_grad()` throughout, not
> `torch.inference_mode()` — documenting the actual state for accuracy, since the
> difference is negligible for this workload either way.

## The `config.yaml` precision modes

> **Superseded (2026-08-21).** This section describes the mode-switch design as
> planned at the time it was written, but `test_video.py` never actually gained a
> `--config` flag or a `yaml` import to read it. The equivalent (and additional)
> tradeoffs were later implemented as unconditional, hardcoded choices directly in
> `test_video.py` instead — see [Hardcoded precision/perf
> choices](#hardcoded-precisionperf-choices-supersedes-configyaml) below for what
> actually shipped. `config.yaml` has since been removed from the repo as dead code.

Steps 1-4 above are unconditional and always bit-exact. Two further optimizations
exist that trade a small amount of numerical accuracy for significant additional
speed. Rather than picking one tradeoff for everyone, `test_video.py` reads a
`precision` setting from `config.yaml` (path overridable with `--config`) with three
levels:

```yaml
precision: exact | tf32 | fast
```

### `exact`

fp32 everywhere, `torch.backends.cuda.matmul.allow_tf32 = False`. Bit-for-bit
identical to the fully-unoptimized script. **3.51 FPS model / 3.04 FPS end-to-end.**

### `tf32` (recommended default)

**Change:** `torch.backends.cuda.matmul.allow_tf32 = True`.

**Why:** cuDNN's own convolutions already default to TF32 on this (Ampere+) GPU —
`torch.backends.cudnn.allow_tf32` is `True` out of the box. `torch.bmm` (used by
`SearchTransfer`'s dense correlation, 40% of per-frame time) goes through cuBLAS, not
cuDNN, and cuBLAS's TF32 flag defaults to `False` separately. This setting just closes
that gap so the correlation step gets the same tensor-core speedup the convs already
have.

**Precision impact measured** on a real frame: max pixel diff **9/255**, only **1.8%**
of pixels touched at all (any channel, by any amount). The affected value (`R_lv3`,
the correlation score used both as an argmax index and as an attention-weight
multiplier `S`) has a wide safety margin here — this is imperceptible in a video.

**Measured effect (full 453-frame video):** model inference 3.51 → 4.02 FPS,
end-to-end 3.04 → 3.41 FPS.

### `fast`

**Change:** additionally wraps the `MainNet` call in `torch.autocast('cuda',
dtype=torch.float16)`.

**Why:** `MainNet` is 55.7% of per-frame time and is pure conv/upsampling work with no
algorithmic redundancy to remove — the only lever left is running it in a faster
numeric format.

**bf16 vs fp16 — tested, not assumed:** the common heuristic is "prefer bf16, it has
fp32's exponent range so it can't overflow" — and that heuristic usually matters most
for architectures like this one, since `MainNet`'s residual blocks have no
BatchNorm/InstanceNorm between them (unlike a classifier ResNet), so activations from
60 stacked residual blocks are exactly the kind of unbounded quantity bf16's wider
range is meant to protect against. So bf16 was tried first. Measuring it against fp32
on a real frame showed the opposite of the expected outcome:

| dtype | max intermediate activation | max pixel diff vs fp32 | mean pixel diff | speed |
|---|---|---|---|---|
| bf16 | 8.7 | **18/255** | 0.237 | 0.1008s/frame |
| fp16 | 8.8 | **2/255** | 0.027 | 0.1008s/frame |

Identical speed (same tensor-core throughput for both 16-bit formats on this GPU), but
fp16 lands ~9x closer to the fp32 baseline. The reason the usual heuristic doesn't
apply to *this checkpoint*: its activations only peak around 8-9 in magnitude, nowhere
near fp16's overflow ceiling (65504) — the overflow risk bf16 guards against never
actually materializes here, so fp16's 3 extra mantissa bits (10 vs bf16's 7) just add
free precision with no corresponding downside. **Conclusion: measure per-checkpoint
rather than assuming the general heuristic — it can point the wrong way.**

**Measured effect:** MainNet alone 0.166s → 0.101s/frame (a 40% cut on the single
largest per-frame cost). Combined with `tf32`: ~5.6 FPS model inference in isolated
benchmarking (~62% faster than `exact`).

**When to use it:** fine for previewing / most video output where a few pixels of
drift on part of the frame is invisible. Avoid it if the output needs to match
`exact` closely (e.g. for a quantitative quality comparison against the original
model).

## Hardcoded precision/perf choices (supersedes `config.yaml`)

The mode-switch design above was replaced with four unconditional changes added
directly to `test_video.py`, no config flag involved. **Benchmarked on a different
clip than the rest of this document — `shourya_bunty_2.mp4` (208 frames), ref
`shourya.png`, same GPU.** Numbers in this section are self-consistent with each
other but not directly comparable to the `ananya.mp4` numbers above (different clip,
resolution, frame count).

### 5. `cudnn.benchmark` on, TF32 left off

**Change:** `torch.backends.cudnn.benchmark = True` unconditionally (same reasoning
as step 3 above). `torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32` were
tried and then deliberately left `False`.

**Why TF32 was rejected here** (unlike the `tf32` config mode above): TF32 truncates
matmul/conv mantissa precision, and `SearchTransfer.bis`'s hard-argmax texture search
is sensitive enough to rounding noise to occasionally flip which reference patch gets
selected for a handful of positions per frame (confirmed by the fp16 experiment in
step 6). TF32 only speeds up ops still running in fp32; once the per-frame loop moved
to fp16 (step 6), the extra speed TF32 could add on top was small relative to the
extra precision risk, so it was left off.

**Methodological finding worth flagging:** the first attempt to measure TF32's impact
used `compare.py` (encode both outputs to `mp4v`, diff the two videos, `--alpha 8` to
amplify) and appeared to show a large, alarming difference (mean 0.77/255 abs diff).
Directly measuring the `mp4v` encode→decode round-trip noise on **identical,
unmodified frames** (re-encode, decode, diff against the untouched in-memory array)
gave **mean 2.58/255** — the codec's own lossy-compression noise is larger than the
numerical effect it was supposedly measuring. **Comparing two independently
re-encoded `mp4v` files is not a reliable way to measure a precision change's true
impact; it has to be measured on raw, uncompressed frames.**

### 6. Mixed precision (fp16) for the per-frame loop only

**Change:** the per-frame loop (`LTE(lrsr)`, the correlation search, `MainNet`) is
wrapped in `torch.autocast('cuda', dtype=torch.float16)`. The one-time reference
feature setup is deliberately **not** autocast: those features are reused by every
frame, so precision lost there is a systematic bias baked into the whole video, not
independent per-frame noise — and that block only costs ~0.3s regardless of
precision, so there's no speed reason to risk it.

**bf16 vs fp16 — tested, not assumed** (same lesson as step `fast` above, reconfirmed
on this checkpoint/clip). Isolated per-frame-loop benchmark, ref features computed in
fp32 for all three:

| dtype | model FPS | speedup vs fp32 | mean pixel diff vs fp32 (raw frames) | worst-case frame diff |
|---|---|---|---|---|
| fp32 (baseline) | 2.50 | 1.0x | — | — |
| bf16 | 5.96 | 2.38x | 0.26 / 255 | 147 / 255 |
| fp16 | 5.87 | 2.35x | **0.05 / 255** | **58 / 255** |

Same speed either way (both use the same tensor-core path), but fp16's 10 mantissa
bits vs bf16's 7 give ~5x less drift and a much smaller worst-case spike, consistent
with step 5's finding that this model's argmax search punishes rounding error. bf16's
wider exponent range buys nothing here since activations never approach fp16's
overflow ceiling. **fp16 was shipped.**

**Measured effect on the real script** (full run, includes preprocessing/write
overhead, before step 8 below): model inference 2.50 → 6.04 FPS (2.4x), end-to-end
4.65 FPS.

### 7. Drop the unused `LTE_copy` VGG19 copy

**Change:** `del model.LTE_copy` right after `load_state_dict`.

**Why:** `TTSR.forward()` only uses `LTE_copy` for the training-time perceptual loss
(the `sr is not None` branch in `model/TTSR.py`), which `test_video.py` never calls
into — it's a second full VGG19-slice copy sitting on the GPU doing nothing during
inference.

**Measured effect:** no change to speed or output (re-run gave 6.04 FPS model / 4.68
FPS end-to-end, matching step 6 within run-to-run noise) — this is pure VRAM headroom
freed up for future work like frame batching, not a throughput change.

### 8. Move the `lr_sr` upsample from CPU (PIL) to GPU (`F.interpolate`)

**Change:** replaced `np.array(Image.fromarray(frame_rgb).resize((lr_w*scale,
lr_h*scale), Image.BICUBIC))` — a per-frame CPU bicubic upsample (e.g. 384px→1536px)
plus a second host→device transfer for the result — with `F.interpolate(lr_t,
scale_factor=scale, mode='bicubic', align_corners=False)` run directly on the
already-uploaded `lr_t` GPU tensor.

**Why:** the CPU PIL resize was single-threaded and re-run from scratch every frame;
doing the same upsample on GPU removes both the CPU cost and one of the two
host→device transfers per frame.

**Caveat:** torch's bicubic kernel isn't bit-identical to PIL's, so this introduces
the same small, tie-flip-driven drift as steps 5-6 — not a bug, a different (also
correct) resampling implementation feeding the search.

**Measured effect:** frame preprocessing 5.03s → 0.14s over 208 frames (**41 FPS →
1534 FPS** for that stage alone, ~37x). Model inference itself is unaffected (6.04 →
6.06 FPS, within noise) since this stage doesn't touch it. End-to-end pipeline: 4.68
→ **5.28 FPS**.

### Summary (this section's clip, `shourya_bunty_2.mp4`)

| State | Model FPS | End-to-end FPS |
|---|---|---|
| fp32, `cudnn.benchmark` on, TF32 off (isolated) | 2.50 | — |
| + fp16 autocast on per-frame loop (step 6) | 6.04 | 4.65 |
| + `LTE_copy` removed (step 7) | 6.04 | 4.68 |
| + GPU-side resize (step 8, current) | 6.06 | **5.28** |

Net: **~2.4x model inference**, **~1.13x further end-to-end** from step 8 alone on
top of step 6/7's gain.

## What was tried and rejected

- **`channels_last` memory format** on `MainNet`: measured *slower*, not faster
  (0.170s → 0.228s/frame). This architecture mixes small-kernel convs with
  `PixelShuffle` upsampling and bicubic interpolation in a way that doesn't benefit
  from the NHWC layout on this GPU/cuDNN version — likely the per-call layout
  conversion cost outweighs any conv speedup. Not used.
- **Batching multiple frames per forward call**: rejected without implementing.
  `SearchTransfer`'s correlation and `MainNet`'s convs both do zero cross-frame
  work-sharing — only the ref-side features (already cached in steps 1-2) are shared
  across frames — so compute and memory both scale linearly with batch size, for no
  reduction in total FLOPs. The only theoretical benefit is amortizing kernel-launch
  overhead, and profiling showed that overhead is negligible on this GPU relative to
  per-op compute time (each frame's ~150 small conv calls take a few milliseconds of
  launch overhead against ~285ms of actual compute). Poor risk/reward; not implemented.

## Files changed

- `test_video.py` — all changes above.
- `config.yaml` — removed; it was never read by `test_video.py` (see the
  superseded-mode note above), so it was dead weight.
- `model/*.py` — **untouched**. Every optimization above works by calling existing
  public submodules (`model.LTE`, `model.SearchTransfer`, `model.MainNet`,
  `model.SearchTransfer.bis`) directly from `test_video.py` instead of going through
  `model.forward()`. This keeps the change scoped to the inference script and away
  from code shared with training (`main.py`, `trainer.py`).

## Summary (`ananya.mp4`, steps 1-4 + planned `config.yaml` modes)

| Mode | Model FPS | End-to-end FPS | Accuracy |
|---|---|---|---|
| Original (unoptimized) | 3.23 | 2.84 | — |
| `exact` (steps 1-4 only) | 3.51 | 3.04 | bit-exact |
| `tf32` (default) | 4.02 | 3.41 | max 9/255 diff, 1.8% of pixels touched |
| `fast` | ~5.6 (isolated) | ~5.0 (est.) | max ~2-9/255 diff on tested frames |

See [step 5-8's summary table](#summary-this-sections-clip-shourya_bunty_2mp4) above
for what actually shipped instead of the `fast`/`tf32` config modes.
