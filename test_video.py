import torch
import numpy as np
import os
import random
import argparse
import sys
import timeit
import cv2
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

model = TTSR.TTSR(base_args).to(device)
load_net = torch.load(param.model, map_location=device)
net_sd = model.state_dict()
net_sd.update(load_net)
model.load_state_dict(net_sd)
model.eval()

print('{} LOADED'.format(param.model))


def to_tensor(img_uint8):
  img = img_uint8.astype(np.float32) / 127.5 - 1.
  return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float().to(device, non_blocking=True)


def tensor_to_uint8(t):
  img = (t + 1.) * 127.5
  img = img.squeeze(0).clamp(0, 255).round().cpu().numpy()
  return np.transpose(img, (1, 2, 0)).astype(np.uint8)


# Load the low-res video
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

print('TEST START\n')

writer = None
model_time_total = 0.0
with torch.no_grad():
  start_t = timeit.default_timer()
  for i, frame_bgr in enumerate(lr_frames_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if (frame_rgb.shape[1], frame_rgb.shape[0]) != (lr_w, lr_h):
      frame_rgb = cv2.resize(frame_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    lr_sr = np.array(Image.fromarray(frame_rgb).resize((lr_w * scale, lr_h * scale), Image.BICUBIC))

    lr_t = to_tensor(frame_rgb)
    lr_sr_t = to_tensor(lr_sr)

    if device == 'cuda':
      torch.cuda.synchronize()
    model_start = timeit.default_timer()
    sr, _, _, _, _ = model(lr=lr_t, lrsr=lr_sr_t, ref=ref_t, refsr=ref_sr_t)
    if device == 'cuda':
      torch.cuda.synchronize()
    model_time_total += timeit.default_timer() - model_start

    sr_frame = tensor_to_uint8(sr)
    sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)

    if writer is None:
      out_h, out_w = sr_frame_bgr.shape[:2]
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      writer = cv2.VideoWriter(param.save_path, fourcc, fps, (out_w, out_h))

    writer.write(sr_frame_bgr)

    if (i + 1) % 10 == 0 or i == total_frames - 1:
      elapsed = timeit.default_timer() - start_t
      print('[TEST] {}/{} frames  {:5.2f}s'.format(i + 1, total_frames, elapsed))

writer.release()
model_fps = total_frames / model_time_total if model_time_total > 0 else float('nan')
print('\nModel-only forward pass: {:.3f}s total, {:.2f} FPS ({} frames)'.format(
    model_time_total, model_fps, total_frames))
print('SAVED SR VIDEO TO: {}'.format(param.save_path))
