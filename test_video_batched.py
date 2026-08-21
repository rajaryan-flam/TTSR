"""Batched TTSR video inference, backed entirely by precompiled TensorRT engines.

Usage: python test_video_batched.py -ref_img <path> -lr_video <path>
Everything else (save path, batch size, engine directory) is a hardcoded
constant below -- edit those directly rather than via CLI.

Requires trt_engines/LTE_<shape>.engine, trt_engines/LTE_ref_<shape>.engine,
and trt_engines/MainNet_<shape>.engine to already exist for this exact
(BATCH_SIZE, -lr_video, -ref_img) combination -- build them first with
convert_tensorrt.py (edit its own BATCH_SIZE/LR_VIDEO/REF_IMG constants to
match, then run `python convert_tensorrt.py`). Engines are fixed-shape/
fixed-batch (no dynamic-shape optimization profile was built), so a mismatch
fails fast with a clear error instead of silently rebuilding or falling back
to eager PyTorch.

Unlike test_video.py (and earlier versions of this script), NO checkpoint
(TTSR.pt) is loaded here. LTE and MainNet both run entirely as TensorRT
engines -- LTE_<shape>.engine for the per-frame/batch lrsr tensor (every
frame) AND LTE_ref_<shape>.engine for the reference image (twice per video),
both with weights already baked in by convert_tensorrt.py's ONNX export. The
only other submodule this script needs is SearchTransfer.bis (pure indexed
tensor movement -- model/SearchTransfer.py's SearchTransfer has zero
learnable parameters, see model/TTSR.py), so it's constructed directly with
no checkpoint. The texture-search glue itself (F.unfold/F.normalize/
torch.matmul/torch.max/SearchTransfer.bis's torch.gather/F.fold) stays eager
PyTorch, matching convert_tensorrt.py's conversion scope.

Precision note: optimizations.md kept the reference-image LTE features in
fp32 throughout this project's history, specifically because they're
computed ONCE and reused for every frame -- precision lost there is a
systematic bias baked into the whole video, not independent per-frame noise
like the fp16/TF32 tradeoffs elsewhere. Running LTE_ref through a
STRONGLY_TYPED fp16 TensorRT engine trades away that fp32 guarantee.
Measured on the real reference image + frame 0 (see convert_tensorrt.py's
ref-LTE precision check): R_lv3_star_arg (the hard-argmax patch selection)
picked a different ref patch at 2.20% of positions -- a real, non-negligible
shift, larger than this codebase's other accepted tradeoffs (TF32: 1.8% of
pixels; step 12's full-graph compile: 0.097%), though still the same
tie-flip *category*, not a qualitative break. Accepted as a deliberate
tradeoff to drop the TTSR.pt dependency; revert to loading TTSR.pt and
calling model.LTE eagerly for the ref features if this drift ever needs to
be ruled out for a given clip.

Every optimization from optimizations.md that still applies to a
TensorRT-engine backend is applied here too: cached/fp16-stored ref-side LTE
features (steps 1/2/9), GPU-side bicubic resize (step 8), cudnn.benchmark on
with TF32 off (step 5), fp16 autocast around the per-batch loop (step 6), and
the overlapped writer thread + CUDA-event timing + GPU-side uint8/BGR
conversion (step 13). `torch.compile` (steps 10/12) does not apply here --
LTE/MainNet run as precompiled TensorRT engines, not eager/compiled PyTorch
modules, so there is nothing for Inductor to compile in this script. Per
optimizations.md's own measurements, that means this script is NOT the
fastest option: TensorRT's MainNet engine benchmarked ~10x slower than
test_video.py's torch.compile'd eager MainNet (no fast kernel for MainNet's
interleaved bicubic resizes), and batching stopped mattering once
torch.compile removed the kernel-launch overhead it would otherwise
amortize. Use this script when you specifically need the TensorRT-engine
path exercised/served; use test_video.py for the fastest wall-clock result.
"""

import torch
import numpy as np
import os
import random
import argparse
import sys
import timeit
import cv2
import queue
import threading
import torch.nn.functional as F
from PIL import Image

from model.SearchTransfer import SearchTransfer
from convert_tensorrt import TRTModule
import warnings
warnings.filterwarnings("ignore")

seed = 216
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ref_img/lr_video are the two inputs that actually change from run to run,
# so they're CLI args, same as test_video.py. Everything else below is a
# fixed-per-deployment constant (output path, batch size, engine directory)
# -- edit those directly instead, mirroring convert_tensorrt.py's own
# BATCH_SIZE constant (keep this script's BATCH_SIZE in sync with whatever
# convert_tensorrt.py was run with, since that's what determines which
# .engine files match).
parser = argparse.ArgumentParser()
parser.add_argument('-ref_img', '--ref_img', type=str, required=True, help='path to the HR reference image')
parser.add_argument('-lr_video', '--lr_video', type=str, required=True, help='path to the low-res input video')
cli_args = parser.parse_args()
REF_IMG = cli_args.ref_img
LR_VIDEO = cli_args.lr_video

SAVE_PATH = '/workspace/outputs/shourya_bunty_trt_sr.mp4'
BATCH_SIZE = 1  # must match a TensorRT engine already built by convert_tensorrt.py for this exact batch_size/lr_video/ref_img
ENGINE_DIR = 'trt_engines'  # directory containing the LTE_*.engine/LTE_ref_*.engine/MainNet_*.engine files produced by convert_tensorrt.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
  sys.exit("TensorRT engines require a CUDA GPU; none is visible to torch.")
scale = 4  # fixed by the pretrained checkpoint's network structure

# Every frame in the video is resized to the same (lr_w, lr_h), so cuDNN's
# per-shape algorithm search (benchmark mode) only pays its one-time cost on
# the first batch and then reuses the fastest kernels for the rest. TF32 is
# left disabled: it truncates matmul/conv mantissa precision, and this
# model's hard-argmax texture search (SearchTransfer.bis) is sensitive
# enough to that rounding noise to occasionally pick a different reference
# patch, producing a visible (if not incorrect) output drift.
torch.backends.cudnn.benchmark = True

# TTSR's texture search does a DENSE correlation between every LR patch and
# every ref patch (at 1/4 resolution): cost/memory scale with
# batch_size * (lr_h/4 * lr_w/4) * (ref_h/4 * ref_w/4). At native size (512px
# LR, ~1000px ref) a single frame already tries to allocate ~65GB. Measured
# safe: both capped at 384px peaks at ~27GB per frame on a 48GB GPU. Batching
# multiplies that per-frame correlation cost by batch_size (the ref-side
# tensors are still shared/broadcast, not duplicated -- see run_batch_model
# below -- but the correlation matrix itself is per-batch-item), so pick
# batch_size relative to your GPU's headroom at whatever max_lr_dim/max_ref_dim
# you use. Frames/ref larger than max_lr_dim/max_ref_dim are downscaled before
# running the network, trading resolution for feasibility.
max_lr_dim = 384
max_ref_dim = 384

save_dir = os.path.dirname(SAVE_PATH)
if save_dir:
  os.makedirs(save_dir, exist_ok=True)

t0 = timeit.default_timer()
# The only eager submodule this script still needs: SearchTransfer.bis is a
# pure batch-index-select (torch.gather under the hood) with zero learnable
# parameters, so it needs no checkpoint -- LTE and MainNet, which DO have
# trained weights, run entirely as TensorRT engines below instead.
search_transfer = SearchTransfer().to(device)
model_load_time = timeit.default_timer() - t0


def to_tensor(img_uint8):
  img = img_uint8.astype(np.float32) / 127.5 - 1.
  return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float().to(device, non_blocking=True)


def frames_to_tensor(frames_uint8):
  # frames_uint8: list of HxWx3 uint8 arrays, all the same shape -> [B,3,H,W]
  batch = np.stack(frames_uint8, axis=0).astype(np.float32) / 127.5 - 1.
  batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
  return torch.from_numpy(batch).float().to(device, non_blocking=True)


def tensor_to_uint8_bgr_batch(t):
  # t: [B,3,H,W] fp16, RGB order, values in [-1,1]. Same reasoning as
  # test_video.py's tensor_to_uint8_bgr: cast to uint8 and swap RGB->BGR on
  # the GPU BEFORE the device->host copy -- halves the transfer (1 byte/pixel
  # vs 2 for fp16) and turns the channel swap into GPU index reordering
  # instead of a per-frame cv2.cvtColor CPU pass. Bit-exact either way --
  # clamp()+round() already guarantee integer values in [0,255].
  img = (t + 1.) * 127.5
  img = img.clamp(0, 255).round().to(torch.uint8)
  img = img[:, [2, 1, 0], :, :]  # RGB -> BGR
  img = img.cpu().numpy()
  return np.transpose(img, (0, 2, 3, 1))  # [B,H,W,3] uint8, BGR


# Load the low-res video
t0 = timeit.default_timer()
cap = cv2.VideoCapture(LR_VIDEO)
if not cap.isOpened():
  raise RuntimeError("Cannot open video: {}".format(LR_VIDEO))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
  fps = 30.0

lr_frames_bgr = []
while True:
  ret, frame = cap.read()
  if not ret:
    break
  lr_frames_bgr.append(frame)
cap.release()
video_load_time = timeit.default_timer() - t0

total_frames = len(lr_frames_bgr)
if total_frames > 0:
  print('VIDEO LOADED: {} frames'.format(total_frames))
else:
  raise RuntimeError("Video is not loaded")

batch_size = max(1, BATCH_SIZE)

# Downscale LR frames to max_lr_dim if needed (memory).
orig_h, orig_w = lr_frames_bgr[0].shape[:2]
lr_ratio = min(1.0, max_lr_dim / max(orig_h, orig_w))
lr_h, lr_w = int(round(orig_h * lr_ratio)), int(round(orig_w * lr_ratio))
if lr_ratio < 1.0:
  print('Downscaling input frames from {}x{} to {}x{} to fit GPU memory (max_lr_dim={})'.format(
      orig_w, orig_h, lr_w, lr_h, max_lr_dim))

# Load the HR reference image, downscale to max_ref_dim if needed (memory),
# then mod-crop to a multiple of `scale`. Unlike ERVSR/C2-Matching, TTSR's
# texture search is resolution-independent between lr and ref, so the ref
# does NOT need to match the LR frame's (upscaled) size.
t0 = timeit.default_timer()
ref_pil = Image.open(REF_IMG).convert('RGB')
ref_w0, ref_h0 = ref_pil.size
ref_ratio = min(1.0, max_ref_dim / max(ref_h0, ref_w0))
if ref_ratio < 1.0:
  ref_pil = ref_pil.resize((int(round(ref_w0 * ref_ratio)), int(round(ref_h0 * ref_ratio))), Image.BICUBIC)
  print('Downscaling reference image from {}x{} to {}x{} to fit GPU memory (max_ref_dim={})'.format(
      ref_w0, ref_h0, ref_pil.size[0], ref_pil.size[1], max_ref_dim))

ref = np.array(ref_pil)
ref_h, ref_w = ref.shape[:2]
ref_h, ref_w = ref_h // scale * scale, ref_w // scale * scale
ref = ref[:ref_h, :ref_w, :]

# Ref_sr: ref degraded the same way the LR frame is (down `scale`x, then back
# up), so the texture search compares like-for-like blur/detail levels.
ref_sr_pil = Image.fromarray(ref).resize((ref_w // scale, ref_h // scale), Image.BICUBIC)
ref_sr = np.array(ref_sr_pil.resize((ref_w, ref_h), Image.BICUBIC))

ref_t = to_tensor(ref)
ref_sr_t = to_tensor(ref_sr)

# Load the precompiled TensorRT engines (built ahead of time by
# convert_tensorrt.py) for LTE at both shapes it's actually called at --
# lte_engine for the per-frame/batch lrsr tensor, lte_ref_engine for the
# reference image -- plus MainNet. Engines are fixed-shape/fixed-batch (no
# dynamic-shape optimization profile was built), so the shape tags below
# must exactly match what convert_tensorrt.py was run with -- same
# batch_size, and the same lr_h/lr_w/ref_h/ref_w that its own
# compute_shapes() derives from LR_VIDEO/REF_IMG (identical downscale logic
# to the blocks above, so matching LR_VIDEO/REF_IMG constants here reproduces
# the same tags). No warmup is needed the way torch.compile needed one: the
# engines are already fully compiled on disk, not lazily compiled on first
# call.
t0 = timeit.default_timer()
shape_tag = '{}x3x{}x{}'.format(batch_size, lr_h, lr_w)
ref_shape_tag = '1x3x{}x{}'.format(ref_h, ref_w)
lte_engine_path = os.path.join(ENGINE_DIR, 'LTE_{}.engine'.format(shape_tag))
lte_ref_engine_path = os.path.join(ENGINE_DIR, 'LTE_ref_{}.engine'.format(ref_shape_tag))
mainnet_engine_path = os.path.join(ENGINE_DIR, 'MainNet_{}.engine'.format(shape_tag))
for engine_path in (lte_engine_path, lte_ref_engine_path, mainnet_engine_path):
  if not os.path.isfile(engine_path):
    raise RuntimeError(
        "Missing TensorRT engine: {}\n"
        "Build it first with convert_tensorrt.py -- set its BATCH_SIZE={}, "
        "LR_VIDEO='{}', REF_IMG='{}' constants to match this run (those are "
        "what determine the '{}'/'{}' shape tags), then run "
        "`python convert_tensorrt.py`.".format(
            engine_path, batch_size, LR_VIDEO, REF_IMG, shape_tag, ref_shape_tag))
lte_engine = TRTModule(lte_engine_path)
lte_ref_engine = TRTModule(lte_ref_engine_path)
mainnet_engine = TRTModule(mainnet_engine_path)
engine_load_time = timeit.default_timer() - t0
print('Loaded TensorRT engines:\n  {}\n  {}\n  {}'.format(lte_engine_path, lte_ref_engine_path, mainnet_engine_path))

# ref/ref_sr are the same on every frame, but TTSR.forward() recomputes their
# LTE (VGG19) features from scratch on every call. Run LTE on them once here
# and reuse the result for the whole video instead of paying for it per
# frame/batch. These stay batch-size-1 -- they're broadcast across whatever
# batch_size the frame loop uses (see run_batch_model), never duplicated.
with torch.no_grad():
  # Runs through lte_ref_engine (STRONGLY_TYPED fp16), not eager fp32 LTE --
  # see this file's module docstring for the measured precision tradeoff
  # (2.20% of R_lv3_star_arg positions flip on the real test clip) that
  # comes with dropping the TTSR.pt checkpoint this way.
  ref_lv1, ref_lv2, ref_lv3 = lte_ref_engine((ref_t.detach() + 1.) / 2.)
  _, _, refsr_lv3 = lte_ref_engine((ref_sr_t.detach() + 1.) / 2.)

  # SearchTransfer.forward() also unfolds/normalizes the ref-side tensors
  # (refsr_lv3, ref_lv1, ref_lv2, ref_lv3) on every call, but those only
  # depend on the ref features cached above -- precompute them once too.
  # (Mirrors model/SearchTransfer.py's forward() exactly; only the lrsr-side
  # unfold there actually varies per frame/batch.)
  #
  # refsr_lv3_unfold drives the argmax search (R_lv3) -- upcast to fp32
  # before unfold/normalize so this cheap, twice-per-video step doesn't
  # compound the fp16 engine's rounding with a second, avoidable rounding
  # pass on top of it (refsr_lv3 itself is already fp16 out of the engine;
  # this can't recover that, only avoid adding to it).
  refsr_lv3_unfold = F.unfold(refsr_lv3.float(), kernel_size=(3, 3), padding=1).permute(0, 2, 1)
  refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
  # ref_lv3/2/1_unfold below are different: SearchTransfer.bis's
  # torch.gather (pure indexed data movement, computed AFTER the argmax is
  # already fixed) and F.fold (a 9-term overlap-add) can't change which
  # patch gets selected, only the value of the patch already selected -- and
  # that value is cast to fp16 one op later anyway when it hits MainNet's
  # first conv under autocast. They're already fp16 straight out of
  # lte_ref_engine, which is exactly the storage step 9 (optimizations.md)
  # already established as safe post-argmax.
  ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
  ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
  ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)
ref_prep_time = timeit.default_timer() - t0


def run_batch_model(lr_t, lr_sr_t):
  # lr_t: [B,3,lr_h,lr_w], lr_sr_t: [B,3,lr_h*scale,lr_w*scale].
  # Everything below is test_video.py's per-frame pipeline with three changes
  # to make it batch-aware and TensorRT-backed, since the ref-side tensors
  # precomputed above have a batch dim of 1 (one shared reference for the
  # whole video):
  #   1. torch.bmm -> torch.matmul, which broadcasts the batch dim (bmm
  #      requires an exact match; matmul doesn't) so refsr_lv3_unfold's
  #      batch-of-1 broadcasts against lrsr_lv3_unfold's batch-of-B.
  #   2. ref_lv3/2/1_unfold are .expand()-ed from batch-1 to batch-B (a view,
  #      no copy -- broadcasts via a stride-0 batch dim) before
  #      SearchTransfer.bis's torch.gather, so gather/fold treat each of the
  #      B items independently instead of silently flattening them together.
  #      (model/SearchTransfer.py itself is untouched; this mirrors its bis()
  #      logic exactly, just called with a batch-expanded `input`.)
  #   3. LTE(...)/MainNet(...) both run as the precompiled TensorRT engines
  #      loaded above -- same math, same fp16 precision (STRONGLY_TYPED
  #      engines built from real .half() weights/inputs, see
  #      convert_tensorrt.py), just running as a precompiled TRT graph
  #      instead of eager/torch.compile'd PyTorch.
  B = lr_t.size(0)
  _, _, lrsr_lv3 = lte_engine((lr_sr_t.detach() + 1.) / 2.)

  lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
  lrsr_lv3_unfold = F.normalize(lrsr_lv3_unfold, dim=1)
  R_lv3 = torch.matmul(refsr_lv3_unfold, lrsr_lv3_unfold)
  R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)

  ref_lv3_unfold_b = ref_lv3_unfold.expand(B, -1, -1)
  ref_lv2_unfold_b = ref_lv2_unfold.expand(B, -1, -1)
  ref_lv1_unfold_b = ref_lv1_unfold.expand(B, -1, -1)
  T_lv3_unfold = search_transfer.bis(ref_lv3_unfold_b, 2, R_lv3_star_arg)
  T_lv2_unfold = search_transfer.bis(ref_lv2_unfold_b, 2, R_lv3_star_arg)
  T_lv1_unfold = search_transfer.bis(ref_lv1_unfold_b, 2, R_lv3_star_arg)

  T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)
  T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2) * 2, lrsr_lv3.size(3) * 2), kernel_size=(6, 6), padding=2, stride=2) / (3. * 3.)
  T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2) * 4, lrsr_lv3.size(3) * 4), kernel_size=(12, 12), padding=4, stride=4) / (3. * 3.)

  S = R_lv3_star.view(B, 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

  return mainnet_engine(lr_t, S, T_lv3, T_lv2, T_lv1)


print('TEST START (batch_size={})\n'.format(batch_size))

writer = None
write_time_total = 0.0
# Frame encode+write runs on a dedicated background thread fed by a bounded
# queue (one queue slot per FRAME, not per batch), so the main thread can
# move straight on to the next batch's GPU work instead of blocking on cv2's
# synchronous mp4v encode -- same reasoning as test_video.py's step 13.
# Single consumer thread preserves frame order via plain FIFO; batches are
# enqueued frame-by-frame in the order they were produced, so cross-batch
# order is preserved too. Sized at 2 full batches of headroom so enqueueing
# one whole batch doesn't immediately block the main thread on the encoder.
write_queue = queue.Queue(maxsize=max(4, batch_size * 2))


def writer_thread_fn():
  global writer, write_time_total
  while True:
    item = write_queue.get()
    if item is None:
      write_queue.task_done()
      break
    w0 = timeit.default_timer()
    if writer is None:
      out_h, out_w = item.shape[:2]
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (out_w, out_h))
    writer.write(item)
    write_time_total += timeit.default_timer() - w0
    write_queue.task_done()


writer_thread = threading.Thread(target=writer_thread_fn, daemon=True)
writer_thread.start()

# Preprocess/model timings used to come from torch.cuda.synchronize() calls
# bracketing each stage -- accurate, but each one blocks the CPU until the
# entire GPU pipeline (including the async TensorRT engine calls) drains,
# which re-serializes every batch regardless of what the write step does and
# defeats the threaded writer above. CUDA events record timestamps on the
# stream without blocking the CPU; querying elapsed_time() does block, so
# query all of them once at the end (after one final synchronize) instead of
# per batch -- same approach as test_video.py's step 13.
preprocess_events = []  # (start, end) per batch
model_events = []       # (end-of-preprocess, end-of-model) per batch
frames_done = 0
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
  start_t = timeit.default_timer()
  for batch_start in range(0, total_frames, batch_size):
    batch_bgr = lr_frames_bgr[batch_start:batch_start + batch_size]
    real_count = len(batch_bgr)

    if device == 'cuda':
      e_pre_start = torch.cuda.Event(enable_timing=True)
      e_pre_end = torch.cuda.Event(enable_timing=True)
      e_model_end = torch.cuda.Event(enable_timing=True)
      e_pre_start.record()

    batch_rgb = []
    for frame_bgr in batch_bgr:
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      if (frame_rgb.shape[1], frame_rgb.shape[0]) != (lr_w, lr_h):
        frame_rgb = cv2.resize(frame_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
      batch_rgb.append(frame_rgb)
    # Pad a short final batch by repeating the last real frame, so every
    # batch that reaches the TensorRT engines has exactly the fixed shape
    # they were built for -- padded outputs are dropped below, before
    # writing.
    while len(batch_rgb) < batch_size:
      batch_rgb.append(batch_rgb[-1])

    lr_t = frames_to_tensor(batch_rgb)
    # GPU bicubic upsample instead of a per-frame CPU PIL resize + a second
    # host->device transfer -- same as test_video.py, just batched.
    lr_sr_t = F.interpolate(lr_t, scale_factor=scale, mode='bicubic', align_corners=False)

    if device == 'cuda':
      e_pre_end.record()

    sr = run_batch_model(lr_t, lr_sr_t)

    if device == 'cuda':
      e_model_end.record()
      preprocess_events.append((e_pre_start, e_pre_end))
      model_events.append((e_pre_end, e_model_end))

    # This device->host copy is the one point per batch that must stay
    # synchronous -- cheap (a handful of ~7MB frames) relative to model
    # inference, so it doesn't meaningfully block anything. Only the slow
    # mp4v *encode* moves to the background thread.
    sr_frames_bgr = tensor_to_uint8_bgr_batch(sr[:real_count])
    for j in range(real_count):
      write_queue.put(sr_frames_bgr[j])

    frames_done += real_count
    if frames_done % 10 < batch_size or frames_done == total_frames:
      elapsed = timeit.default_timer() - start_t
      print('[TEST] {}/{} frames  {:5.2f}s'.format(frames_done, total_frames, elapsed))

write_queue.put(None)
writer_thread.join()
writer.release()
loop_time_total = timeit.default_timer() - start_t

if device == 'cuda':
  torch.cuda.synchronize()
  preprocess_time_total = sum(s.elapsed_time(e) for s, e in preprocess_events) / 1000.0
  model_time_total = sum(s.elapsed_time(e) for s, e in model_events) / 1000.0
else:
  preprocess_time_total = model_time_total = 0.0

total_time = model_load_time + video_load_time + ref_prep_time + engine_load_time + loop_time_total
model_fps = total_frames / model_time_total if model_time_total > 0 else float('nan')
pipeline_fps = total_frames / loop_time_total if loop_time_total > 0 else float('nan')

print('\n===== TIMING BREAKDOWN ({} frames, batch_size={}) ====='.format(total_frames, batch_size))
print('Module init:        {:7.3f}s'.format(model_load_time))
print('Video load:        {:7.3f}s'.format(video_load_time))
print('Engine load:       {:7.3f}s'.format(engine_load_time))
print('Ref image prep:    {:7.3f}s'.format(ref_prep_time))
print('Frame preprocess:  {:7.3f}s  ({:.2f} FPS)  [via CUDA events, not a blocking sync]'.format(preprocess_time_total, total_frames / preprocess_time_total if preprocess_time_total > 0 else float('nan')))
print('Model inference:   {:7.3f}s  ({:.2f} FPS)  [via CUDA events, not a blocking sync]'.format(model_time_total, model_fps))
print('Frame write:       {:7.3f}s  ({:.2f} FPS)  [writer-thread busy time -- overlaps with the next batch\'s GPU work above, not additive with it]'.format(write_time_total, total_frames / write_time_total if write_time_total > 0 else float('nan')))
print('-----')
print('Per-frame loop total: {:7.3f}s  ({:.2f} FPS end-to-end)'.format(loop_time_total, pipeline_fps))
print('Grand total runtime:  {:7.3f}s'.format(total_time))
print('SAVED SR VIDEO TO: {}'.format(SAVE_PATH))
