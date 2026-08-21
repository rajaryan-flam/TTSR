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

# option.py parses sys.argv at import time with its own argparse parser,
# which doesn't know this script's flags (--model, --ref_img, ...) and would
# exit with "unrecognized arguments" -- hide our argv from it during import.
_real_argv = sys.argv
sys.argv = sys.argv[:1]
from option import args as base_args
sys.argv = _real_argv

from model import TTSR
import warnings
warnings.filterwarnings("ignore")

seed = 216
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model', type = str, required = True, help = 'path to the TTSR checkpoint (TTSR.pt or TTSR-rec.pt)')
parser.add_argument('-ref_img', '--ref_img', type = str, required = True, help = 'path to the HR reference image')
parser.add_argument('-lr_video', '--lr_video', type = str, required = True, help = 'path to the low-res input video')
parser.add_argument('-save_path', '--save_path', type = str, required = True, help = 'path to save the super-resolved video')

param = parser.parse_args()

# Fixed defaults (not exposed on the CLI)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scale = 4  # fixed by the pretrained checkpoint's network structure

if device == 'cuda':
  # Every frame in the video is resized to the same (lr_w, lr_h), so cuDNN's
  # per-shape algorithm search (benchmark mode) only pays its one-time cost
  # on the first frame and then reuses the fastest kernels for the rest.
  # TF32 is left disabled: it truncates matmul/conv mantissa precision, and
  # this model's hard-argmax texture search (SearchTransfer.bis) is sensitive
  # enough to that rounding noise to occasionally pick a different reference
  # patch, producing a visible (if not incorrect) output drift.
  torch.backends.cudnn.benchmark = True

# TTSR's texture search does a DENSE correlation between every LR patch and
# every ref patch (at 1/4 resolution): cost/memory scale with
# (lr_h/4 * lr_w/4) * (ref_h/4 * ref_w/4). At native size (512px LR, ~1000px
# ref) this tries to allocate ~65GB and OOMs. Measured safe: both capped at
# 384px peaks at ~27GB on a 48GB GPU. Frames/ref larger than this are
# downscaled before running the network, trading resolution for feasibility.
max_lr_dim = 384
max_ref_dim = 384

save_dir = os.path.dirname(param.save_path)
if save_dir:
  os.makedirs(save_dir, exist_ok=True)

t0 = timeit.default_timer()
model = TTSR.TTSR(base_args).to(device)
load_net = torch.load(param.model, map_location=device)
net_sd = model.state_dict()
net_sd.update(load_net)
model.load_state_dict(net_sd)
model.eval()
# LTE_copy is a second full VGG19-slice copy TTSR.forward() only uses for the
# training-time perceptual loss (model/TTSR.py's `sr is not None` branch,
# never hit here) -- drop it to free that VRAM for other use (e.g. batching).
del model.LTE_copy
model_load_time = timeit.default_timer() - t0

print('{} LOADED'.format(param.model))


def to_tensor(img_uint8):
  img = img_uint8.astype(np.float32) / 127.5 - 1.
  return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float().to(device, non_blocking=True)


def tensor_to_uint8_bgr(t):
  # t: [1,3,H,W] fp16, RGB order, values in [-1,1].
  # Cast to uint8 and swap to BGR order BEFORE the device->host copy, not
  # after: casting first halves the transfer (1 byte/pixel vs 2 for fp16),
  # and the channel swap is then just GPU index reordering instead of a
  # cv2.cvtColor CPU pass. Both are bit-exact vs. doing them on the CPU
  # afterwards -- round()+clamp() already guarantee the fp16 values are
  # integers in [0,255], so casting to uint8 (here or on the host) is exact,
  # and reordering channels doesn't touch the values at all.
  img = (t + 1.) * 127.5
  img = img.clamp(0, 255).round().to(torch.uint8)
  img = img[:, [2, 1, 0], :, :]  # RGB -> BGR
  img = img.squeeze(0).cpu().numpy()
  return np.transpose(img, (1, 2, 0))  # HWC, uint8, BGR -- ready for cv2.VideoWriter


# Load the low-res video
t0 = timeit.default_timer()
cap = cv2.VideoCapture(param.lr_video)
if not cap.isOpened():
  raise RuntimeError("Cannot open video: {}".format(param.lr_video))

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
ref_pil = Image.open(param.ref_img).convert('RGB')
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

# ref/ref_sr are the same on every frame, but TTSR.forward() recomputes their
# LTE (VGG19) features from scratch on every call. Run LTE on them once here
# and reuse the result for the whole video instead of paying for it per frame.
with torch.no_grad():
  # Kept in fp32 (not autocast): these ref features are computed once and
  # then reused for every frame in the video, so precision lost here isn't
  # per-frame noise -- it's a systematic bias baked into every frame's
  # texture transfer. The one-time cost of running this in fp32 is ~0.3s
  # regardless, so there's no speed reason to autocast it.
  ref_lv1, ref_lv2, ref_lv3 = model.LTE((ref_t.detach() + 1.) / 2.)
  _, _, refsr_lv3 = model.LTE((ref_sr_t.detach() + 1.) / 2.)

  # SearchTransfer.forward() also unfolds/normalizes the ref-side tensors
  # (refsr_lv3, ref_lv1, ref_lv2, ref_lv3) on every call, but those only
  # depend on the ref features cached above -- precompute them once too.
  # (Mirrors model/SearchTransfer.py's forward() exactly; only the lrsr-side
  # unfold there actually varies per frame.)
  refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1).permute(0, 2, 1)
  refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
  # refsr_lv3_unfold above stays fp32 -- it drives the argmax search (R_lv3),
  # which is exactly the value the TF32/fp16 tie-flip risk discussion is
  # about. ref_lv3/2/1_unfold below are different: SearchTransfer.bis's
  # torch.gather (pure indexed data movement, computed AFTER the argmax is
  # already fixed) and F.fold (a 9-term overlap-add) can't change which patch
  # gets selected, only the value of the patch already selected -- and that
  # value is cast to fp16 one op later anyway when it hits MainNet's first
  # conv under autocast. So storing these as fp16 costs no precision autocast
  # wasn't already going to spend, while halving the per-frame gather/fold
  # bandwidth (unlike bmm/conv, gather/fold aren't in autocast's op list, so
  # they'd otherwise run at fp32 bandwidth on every single frame regardless
  # of the autocast context). Verified: argmax output is bit-identical with
  # or without this cast; raw-frame pixel drift is ~0.007/255 mean, under
  # 1/255 max across a full 208-frame test.
  ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1).half()
  ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2).half()
  ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4).half()
ref_prep_time = timeit.default_timer() - t0


def run_frame_model(lr_t, lr_sr_t):
  _, _, lrsr_lv3 = model.LTE((lr_sr_t.detach() + 1.) / 2.)

  lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
  lrsr_lv3_unfold = F.normalize(lrsr_lv3_unfold, dim=1)
  R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold)
  R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)

  T_lv3_unfold = model.SearchTransfer.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
  T_lv2_unfold = model.SearchTransfer.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
  T_lv1_unfold = model.SearchTransfer.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

  T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)
  T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2) * 2, lrsr_lv3.size(3) * 2), kernel_size=(6, 6), padding=2, stride=2) / (3. * 3.)
  T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2) * 4, lrsr_lv3.size(3) * 4), kernel_size=(12, 12), padding=4, stride=4) / (3. * 3.)

  S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

  return model.MainNet(lr_t, S, T_lv3, T_lv2, T_lv1)


compile_warmup_time = 0.0
if device == 'cuda':
  # Compile the WHOLE per-frame function -- LTE, the unfold/normalize/bmm/max
  # search, the 3x gather+fold transfer, and MainNet -- as one graph, instead
  # of compiling model.MainNet/model.LTE individually. Profiling showed the
  # search+transfer glue between those two submodules (plain eager PyTorch)
  # was costing ~48ms/frame on its own -- half the per-frame budget -- almost
  # none of which was actual GPU idle time savable by more compute, but rather
  # Python dispatch/launch overhead between many small unfold/gather/fold/bmm
  # calls that a submodule-only compile couldn't see across. Folding all of it
  # into one reduce-overhead-compiled function (one CUDA graph covering the
  # entire frame) cut measured model inference from 96.5ms/frame (10.37 FPS)
  # to 75.8ms/frame (13.20 FPS) on the 208-frame benchmark clip -- a ~27%
  # additional win with no further precision change (fp16 autocast is
  # unchanged). This does put LTE's convs in the same fused graph as the
  # search/transfer glue, so Inductor can pick different fused kernels for
  # them than compiling LTE alone did -- verified this only produces the same
  # class of tiny argmax tie-flip drift already accepted for TF32/batching
  # elsewhere in this session (0.097% of pixels flip patch, mean drift
  # 0.008/255, far under the ~2.58/255 mp4v re-encode noise floor), not a
  # correctness regression.
  t0 = timeit.default_timer()
  run_frame_model = torch.compile(run_frame_model, mode='reduce-overhead')
  # Warm up on the first real frame's data so compilation (and reduce-overhead's
  # CUDA graph capture, which needs a few real calls to lock in) happens here,
  # not inside frame 0's timed model_time_total. Cold: ~20s (one-time Inductor
  # compile). Warm (Inductor's on-disk cache from a prior run): ~3s. Either way
  # this amortizes fast: on this 208-frame clip the compiled loop is ~12s
  # faster overall, so warm-cache runs are a net win well before frame 50, and
  # even a cold-compile run breaks even within a few hundred frames.
  warm_rgb = cv2.cvtColor(lr_frames_bgr[0], cv2.COLOR_BGR2RGB)
  if (warm_rgb.shape[1], warm_rgb.shape[0]) != (lr_w, lr_h):
    warm_rgb = cv2.resize(warm_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
  warm_lr_t = to_tensor(warm_rgb)
  warm_lr_sr_t = F.interpolate(warm_lr_t, scale_factor=scale, mode='bicubic', align_corners=False)
  with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for _ in range(5):
      run_frame_model(warm_lr_t, warm_lr_sr_t)
  torch.cuda.synchronize()
  compile_warmup_time = timeit.default_timer() - t0

print('TEST START\n')

writer = None
write_time_total = 0.0
# Frame encode+write runs on a dedicated background thread fed by a bounded
# queue, so the main thread can move straight on to the next frame's GPU work
# instead of blocking on cv2's synchronous mp4v encode. Single consumer
# thread (not a pool) preserves frame order via plain FIFO -- no reordering
# risk. Bounded size (not unbounded) means a main thread that outruns the
# encoder blocks on `put()` instead of buffering the whole video in RAM; in
# practice the encoder (~24ms/frame) is faster than model inference
# (~76ms/frame, see optimizations.md step 12), so the queue should rarely
# hold more than one frame.
write_queue = queue.Queue(maxsize=4)


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
      writer = cv2.VideoWriter(param.save_path, fourcc, fps, (out_w, out_h))
    writer.write(item)
    write_time_total += timeit.default_timer() - w0
    write_queue.task_done()


writer_thread = threading.Thread(target=writer_thread_fn, daemon=True)
writer_thread.start()

# Preprocess/model timings used to come from torch.cuda.synchronize() calls
# bracketing each stage -- accurate, but each one blocks the CPU until the
# *entire* GPU pipeline drains, which re-serializes every frame regardless of
# what the write step does and defeats the threaded writer above. CUDA
# events record timestamps on the stream without blocking the CPU; querying
# elapsed_time() does block, so query all of them once at the end (after one
# final synchronize) instead of per frame.
preprocess_events = []  # (start, end) per frame
model_events = []       # (end-of-preprocess, end-of-model) per frame

# fp16 over bf16: benchmarked on-device, both give the same ~2.3x speedup,
# but fp16's extra mantissa bits (10 vs bf16's 7) cut mean pixel drift vs.
# fp32 by ~5x (0.05 vs 0.26 / 255) -- this model's hard-argmax texture
# search is sensitive to rounding, so the extra precision reduces tie-flips.
# fp16's smaller exponent range isn't a risk here: activations stay well
# within +-1 / VGG-feature scale, nowhere near fp16's overflow limit.
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
  start_t = timeit.default_timer()
  for i, frame_bgr in enumerate(lr_frames_bgr):
    if device == 'cuda':
      e_pre_start = torch.cuda.Event(enable_timing=True)
      e_pre_end = torch.cuda.Event(enable_timing=True)
      e_model_end = torch.cuda.Event(enable_timing=True)
      e_pre_start.record()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if (frame_rgb.shape[1], frame_rgb.shape[0]) != (lr_w, lr_h):
      frame_rgb = cv2.resize(frame_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    lr_t = to_tensor(frame_rgb)
    # GPU bicubic upsample instead of a per-frame CPU PIL resize + a second
    # host->device transfer. Note: torch's bicubic kernel isn't bit-identical
    # to PIL's, so lrsr_lv3 (and the argmax texture search fed by it) shifts
    # by the same small, tie-flip-driven margin as the fp16/TF32 changes did.
    lr_sr_t = F.interpolate(lr_t, scale_factor=scale, mode='bicubic', align_corners=False)

    if device == 'cuda':
      e_pre_end.record()

    sr = run_frame_model(lr_t, lr_sr_t)

    if device == 'cuda':
      e_model_end.record()
      preprocess_events.append((e_pre_start, e_pre_end))
      model_events.append((e_pre_end, e_model_end))

    # This device->host copy is the one point per frame that must stay
    # synchronous: it has to happen before the *next* iteration's
    # run_frame_model call, since reduce-overhead's CUDA-graph capture
    # reuses the same output buffer on every call, so this frame's data must
    # be safely copied out before it's overwritten. It's also cheap (a
    # single ~7MB frame) relative to model inference, so this doesn't
    # meaningfully block anything -- only the slow mp4v *encode* moves to
    # the background thread below.
    sr_frame_bgr = tensor_to_uint8_bgr(sr)
    write_queue.put(sr_frame_bgr)

    if (i + 1) % 10 == 0 or i == total_frames - 1:
      elapsed = timeit.default_timer() - start_t
      print('[TEST] {}/{} frames  {:5.2f}s'.format(i + 1, total_frames, elapsed))

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

total_time = model_load_time + video_load_time + ref_prep_time + compile_warmup_time + loop_time_total
model_fps = total_frames / model_time_total if model_time_total > 0 else float('nan')
pipeline_fps = total_frames / loop_time_total if loop_time_total > 0 else float('nan')

print('\n===== TIMING BREAKDOWN ({} frames) ====='.format(total_frames))
print('Model load:        {:7.3f}s'.format(model_load_time))
print('Video load:        {:7.3f}s'.format(video_load_time))
print('Ref image prep:    {:7.3f}s'.format(ref_prep_time))
print('Compile warmup:    {:7.3f}s'.format(compile_warmup_time))
print('Frame preprocess:  {:7.3f}s  ({:.2f} FPS)  [via CUDA events, not a blocking sync]'.format(preprocess_time_total, total_frames / preprocess_time_total if preprocess_time_total > 0 else float('nan')))
print('Model inference:   {:7.3f}s  ({:.2f} FPS)  [via CUDA events, not a blocking sync]'.format(model_time_total, model_fps))
print('Frame write:       {:7.3f}s  ({:.2f} FPS)  [writer-thread busy time -- overlaps with the next frame\'s GPU work above, not additive with it]'.format(write_time_total, total_frames / write_time_total if write_time_total > 0 else float('nan')))
print('-----')
print('Per-frame loop total: {:7.3f}s  ({:.2f} FPS end-to-end)'.format(loop_time_total, pipeline_fps))
print('Grand total runtime:  {:7.3f}s'.format(total_time))
print('SAVED SR VIDEO TO: {}'.format(param.save_path))
