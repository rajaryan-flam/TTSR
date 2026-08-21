"""Convert TTSR's MainNet and LTE submodules to TensorRT .engine files.

Scope: MainNet and LTE are converted -- the same two submodules already
proven (in test_video.py) to be clean torch.compile targets, because they're
plain conv/relu/PixelShuffle/bicubic-interpolate networks with a fixed shape
for the whole video and zero data-dependent branching. The custom texture-
search glue in test_video.py's run_frame_model() (F.unfold/F.normalize/
torch.bmm/torch.max/SearchTransfer.bis's torch.gather/F.fold) stays in eager
PyTorch: it's already fast (see optimizations.md steps 9-10) and doesn't map
onto a single static ONNX/TensorRT graph as cleanly as two independent
conv-only networks do, so converting it would add engineering risk for little
remaining upside.

LTE is exported TWICE, as two separate engines with two separate shapes: once
at the per-frame/batch lrsr shape (LTE_<batch>x3x<lr_h>x<lr_w>.engine, runs
every frame) and once at the reference-image's own shape
(LTE_ref_1x3x<ref_h>x<ref_w>.engine, runs only twice per video, batch always
1). The second one exists so a caller can run the reference-image LTE calls
through TensorRT too instead of needing TTSR.pt loaded for eager fp32
weights -- see main()'s ref-LTE precision check (compares fp32 eager output
against this engine's fp16 output on the real reference image and reports
whether the hard-argmax texture search actually shifts) for whether that
tradeoff is safe to ship, since the reference features were deliberately
kept fp32 throughout this project's history (a one-time systematic bias
risk, not per-frame noise -- see optimizations.md).

Pipeline per submodule: PyTorch -> ONNX (torch.onnx.export, exported in true
fp16) -> TensorRT engine (TensorRT's own Python API: Builder + OnnxParser +
BuilderConfig), built as a STRONGLY_TYPED network so precision comes from the
ONNX graph's own fp16 tensors -- TensorRT 10+ removed the old "prefer fp16"
builder flag this script originally tried to use. This still matches the
precision decision established in test_video.py (this model's hard-argmax
texture search is sensitive to TF32-style rounding noise; fp16 was
benchmarked as strictly more accurate than bf16 at the same speed). A smoke
test after each successful build runs both the original PyTorch module and
the engine on the same random input and reports the max/mean difference, so a
broken conversion fails loudly here instead of silently corrupting output
later.

MainNet's Myelin backend fuses essentially the ENTIRE network (from the first
Resize through MergeTail's final Clip) into one opaque compiled kernel, which
genuinely needs a large builder workspace to even attempt compiling --
verbose TensorRT logging (trt.Logger.VERBOSE) showed a real, fixed ~51GB
requirement ("Need 51432728576" bytes), not something that scales with
whatever workspace you happen to give it. WORKSPACE_GB below is set well
above that; drop it and MainNet's build will fail with a "Could not find any
implementation" error that looks like an op-support problem but is really
just workspace starvation for this one big fused node. Both LTE and MainNet
build and pass their smoke tests at WORKSPACE_GB=60 on this 96GB GPU.

Engines are shape- and batch-size-specific (TensorRT does not support
loading an engine built for one input shape and feeding it another, unless
built with a dynamic-shape optimization profile, which this script does not
use -- keep it simple, one engine per shape). Point LR_VIDEO/REF_IMG below at
whatever you'll actually run inference on, so the exported shapes match
exactly; the output filenames encode the shape so multiple engines for
different videos/batch sizes don't collide.

Requires: pip install tensorrt onnx (both must be present; this environment's
installed CUDA/driver must also be new enough for whatever TensorRT version
you install -- see /etc/vast_agents' CUDA compatibility notes if unsure).

No CLI -- edit the constants below and run: python convert_tensorrt.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

_real_argv = sys.argv
sys.argv = sys.argv[:1]
from option import args as base_args
sys.argv = _real_argv

from model import TTSR
import warnings
warnings.filterwarnings("ignore")

try:
  import tensorrt as trt
except ImportError:
  sys.exit(
      "tensorrt is not installed in this Python environment.\n"
      "Install it with: pip install tensorrt\n"
      "(A matching NVIDIA driver/CUDA toolkit is also required -- see\n"
      "/etc/vast_agents' CUDA compatibility notes if the install fails.)")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Edit these directly instead of passing CLI args.
MODEL_PATH = 'TTSR.pt'
REF_IMG = '/workspace/highres_images/shourya.png'
LR_VIDEO = '/workspace/lowres/shourya_bunty_2.mp4'
BATCH_SIZE = 1
OUTPUT_DIR = 'trt_engines'
OPSET = 17
WORKSPACE_GB = 60.0


def compute_shapes(lr_video, ref_img):
  # Mirrors test_video.py's shape derivation exactly, so the engine's fixed
  # input shape matches what test_video.py would actually feed it for this
  # video/ref pair.
  scale = 4
  max_lr_dim = 384
  max_ref_dim = 384

  cap = cv2.VideoCapture(lr_video)
  if not cap.isOpened():
    raise RuntimeError("Cannot open video: {}".format(lr_video))
  ret, frame = cap.read()
  cap.release()
  if not ret:
    raise RuntimeError("Video has no frames: {}".format(lr_video))
  orig_h, orig_w = frame.shape[:2]
  lr_ratio = min(1.0, max_lr_dim / max(orig_h, orig_w))
  lr_h, lr_w = int(round(orig_h * lr_ratio)), int(round(orig_w * lr_ratio))

  ref_pil = Image.open(ref_img).convert('RGB')
  ref_w0, ref_h0 = ref_pil.size
  ref_ratio = min(1.0, max_ref_dim / max(ref_h0, ref_w0))
  ref_w1 = int(round(ref_w0 * ref_ratio)) if ref_ratio < 1.0 else ref_w0
  ref_h1 = int(round(ref_h0 * ref_ratio)) if ref_ratio < 1.0 else ref_h0
  ref_h, ref_w = ref_h1 // scale * scale, ref_w1 // scale * scale

  return lr_h, lr_w, ref_h, ref_w, scale


def load_ref_tensors(ref_img, max_ref_dim, scale, device):
  # Reproduces test_video_batched.py's reference-image preprocessing exactly
  # (downscale to max_ref_dim, mod-crop to a multiple of scale, then degrade
  # ref by scale x and upsample back for ref_sr) so the precision check below
  # runs on the actual tensors that script feeds to LTE, not synthetic noise.
  ref_pil = Image.open(ref_img).convert('RGB')
  ref_w0, ref_h0 = ref_pil.size
  ref_ratio = min(1.0, max_ref_dim / max(ref_h0, ref_w0))
  if ref_ratio < 1.0:
    ref_pil = ref_pil.resize((int(round(ref_w0 * ref_ratio)), int(round(ref_h0 * ref_ratio))), Image.BICUBIC)
  ref = np.array(ref_pil)
  ref_h, ref_w = ref.shape[:2]
  ref_h, ref_w = ref_h // scale * scale, ref_w // scale * scale
  ref = ref[:ref_h, :ref_w, :]

  ref_sr_pil = Image.fromarray(ref).resize((ref_w // scale, ref_h // scale), Image.BICUBIC)
  ref_sr = np.array(ref_sr_pil.resize((ref_w, ref_h), Image.BICUBIC))

  def to_tensor(img_uint8):
    img = img_uint8.astype(np.float32) / 127.5 - 1.
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float().to(device)

  return to_tensor(ref), to_tensor(ref_sr)


def load_first_frame_lrsr(lr_video, lr_h, lr_w, scale, device):
  # Reproduces test_video_batched.py's per-frame preprocessing for frame 0
  # (BGR->RGB, resize to lr_h x lr_w, GPU bicubic upsample by scale) so the
  # argmax check below compares against a real lrsr_lv3, not synthetic noise.
  cap = cv2.VideoCapture(lr_video)
  ret, frame_bgr = cap.read()
  cap.release()
  if not ret:
    raise RuntimeError("Video has no frames: {}".format(lr_video))
  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  if (frame_rgb.shape[1], frame_rgb.shape[0]) != (lr_w, lr_h):
    frame_rgb = cv2.resize(frame_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
  img = frame_rgb.astype(np.float32) / 127.5 - 1.
  lr_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float().to(device)
  return F.interpolate(lr_t, scale_factor=scale, mode='bicubic', align_corners=False)


def check_ref_lte_precision(ref_lv_fp32, ref_lv_trt, lrsr_lv3, names):
  # optimizations.md's own stated reason for keeping ref-feature computation
  # in fp32 throughout its history: those features are computed ONCE and
  # reused for every frame, so precision lost there is a systematic bias
  # baked into the whole video, not independent per-frame noise -- a
  # different, stricter risk than the per-frame fp16 autocast already
  # validated elsewhere. This checks whether switching the reference-image
  # LTE call to the STRONGLY_TYPED fp16 TensorRT engine actually triggers
  # that risk, using the REAL reference image (via load_ref_tensors) and a
  # REAL frame (via load_first_frame_lrsr) -- not random dummy input like
  # smoke_test above -- and specifically checks the one thing that matters:
  # whether R_lv3_star_arg (the hard-argmax texture-patch selection, the
  # exact quantity optimizations.md's TF32/fp16 rejections were guarding)
  # changes, not just whether raw feature values drift.
  print('\n[ref-LTE precision check] fp32 eager vs. fp16 TensorRT engine, on the REAL reference image:')
  for name, fp32_t, trt_t in zip(names, ref_lv_fp32, ref_lv_trt):
    diff = (fp32_t.float() - trt_t.float()).abs()
    print('  {}: mean abs diff={:.6f}  max abs diff={:.6f}'.format(name, diff.mean().item(), diff.max().item()))

  refsr_lv3_fp32, refsr_lv3_trt = ref_lv_fp32[-1], ref_lv_trt[-1]

  def argmax_from(refsr_lv3):
    refsr_unfold = F.unfold(refsr_lv3.float(), kernel_size=(3, 3), padding=1).permute(0, 2, 1)
    refsr_unfold = F.normalize(refsr_unfold, dim=2)
    lrsr_unfold = F.unfold(lrsr_lv3.float(), kernel_size=(3, 3), padding=1)
    lrsr_unfold = F.normalize(lrsr_unfold, dim=1)
    R_lv3 = torch.matmul(refsr_unfold, lrsr_unfold)
    return torch.max(R_lv3, dim=1)[1]

  arg_fp32 = argmax_from(refsr_lv3_fp32)
  arg_trt = argmax_from(refsr_lv3_trt)
  mismatch = (arg_fp32 != arg_trt).float().mean().item()
  print('  R_lv3_star_arg mismatch on frame 0: {:.4%} of positions picked a different ref patch'.format(mismatch))


def export_onnx_lte(model, batch_size, h, w, onnx_path, opset, input_name='x'):
  # LTE is exported once per distinct input shape it's actually called at.
  # Two shapes matter: the per-frame/batch lrsr tensor (h=lr_h*scale,
  # w=lr_w*scale, runs once per frame/batch) and the reference-image tensor
  # (h=ref_h, w=ref_w, batch_size=1, runs only twice total per video) -- see
  # main() below for both call sites.
  #
  # Exported in true fp16 (module weights AND inputs cast with .half()), not
  # test_video.py's fp32-weights-plus-autocast mixed precision: TensorRT 10+
  # removed the old "prefer fp16" builder flag in favor of strongly-typed
  # networks, where precision comes directly from the ONNX graph's own
  # declared tensor dtypes (see build_engine below). This fp16 policy is
  # already validated as safe for the per-frame path (optimizations.md steps
  # 6/10). It is NOT yet validated for the reference-image path -- that path
  # was deliberately kept fp32 throughout optimizations.md's history, since
  # precision lost there is a one-time systematic bias baked into every
  # frame's texture transfer, not independent per-frame noise. See main()'s
  # ref-LTE precision check, run against the real reference image, for
  # whether that risk actually materializes here.
  model.LTE.half()
  dummy = torch.randn(batch_size, 3, h, w, device='cuda', dtype=torch.float16)
  torch.onnx.export(
      model.LTE, dummy, onnx_path,
      input_names=[input_name], output_names=['lv1', 'lv2', 'lv3'],
      opset_version=opset, do_constant_folding=True, dynamo=False)
  return dummy


def export_onnx_mainnet(model, batch_size, lr_h, lr_w, n_feats, onnx_path, opset):
  model.MainNet.half()
  lr_t = torch.randn(batch_size, 3, lr_h, lr_w, device='cuda', dtype=torch.float16)
  S = torch.rand(batch_size, 1, lr_h, lr_w, device='cuda', dtype=torch.float16)
  T_lv3 = torch.randn(batch_size, 256, lr_h, lr_w, device='cuda', dtype=torch.float16)
  T_lv2 = torch.randn(batch_size, 128, lr_h * 2, lr_w * 2, device='cuda', dtype=torch.float16)
  T_lv1 = torch.randn(batch_size, 64, lr_h * 4, lr_w * 4, device='cuda', dtype=torch.float16)
  torch.onnx.export(
      model.MainNet, (lr_t, S, T_lv3, T_lv2, T_lv1), onnx_path,
      input_names=['lr', 'S', 'T_lv3', 'T_lv2', 'T_lv1'], output_names=['sr'],
      opset_version=opset, do_constant_folding=True, dynamo=False)
  return (lr_t, S, T_lv3, T_lv2, T_lv1)


def build_engine(onnx_path, engine_path, workspace_gb):
  builder = trt.Builder(TRT_LOGGER)
  # TensorRT 10+ removed implicit-batch mode entirely (create_network() with
  # no flags is already explicit-batch, the only mode left) and also removed
  # BuilderFlag.FP16/TF32 as ways to hint a preferred precision. The modern
  # replacement is STRONGLY_TYPED: the network's precision is taken directly
  # from the tensor dtypes already baked into the ONNX graph (fp16 here,
  # since export_onnx_lte/export_onnx_mainnet exported real fp16 tensors)
  # rather than a builder-level global preference with fp32 fallback.
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
  parser = trt.OnnxParser(network, TRT_LOGGER)

  with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
      errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
      raise RuntimeError("Failed to parse {}:\n{}".format(onnx_path, errors))

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
  # No FP16/TF32 builder flags needed (or available) here -- STRONGLY_TYPED
  # above already commits the network to the fp16 types declared in the ONNX
  # graph, matching test_video.py's established precision decision: fp16 was
  # benchmarked as strictly more accurate than bf16 at the same speed, and
  # this model's hard-argmax texture search is sensitive to TF32-style
  # mantissa truncation -- a strongly-typed fp16 network has no TF32 fallback
  # path to worry about in the first place.
  #
  # MainNet's Myelin backend fuses the ENTIRE network (from the first Resize
  # through MergeTail's final Clip -- basically all of it) into one opaque
  # kernel, which genuinely needs a large workspace to compile: verbose
  # builder logging (trt.Logger.VERBOSE) revealed "Exceeded mem budget of
  # 8602283142. Need 51432728576" -- a real, fixed ~51GB requirement, not a
  # runaway estimate that scales with whatever workspace you give it (an
  # earlier, wrong theory formed from only checking the WARNING-level output,
  # which just echoes back the configured budget rather than the real need).
  # WORKSPACE_GB above must stay comfortably north of that for MainNet to
  # build at all.

  serialized_engine = builder.build_serialized_network(network, config)
  if serialized_engine is None:
    raise RuntimeError("TensorRT engine build failed for {}".format(onnx_path))

  with open(engine_path, 'wb') as f:
    f.write(serialized_engine)
  return serialized_engine


class TRTModule:
  """Minimal wrapper to run a serialized TensorRT engine on torch CUDA tensors."""

  def __init__(self, engine_path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
      self.engine = runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()
    self.input_names = []
    self.output_names = []
    for i in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(i)
      if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        self.input_names.append(name)
      else:
        self.output_names.append(name)

  def __call__(self, *inputs):
    assert len(inputs) == len(self.input_names), \
        "expected {} inputs, got {}".format(len(self.input_names), len(inputs))
    inputs = [t.contiguous().float().half() if t.dtype != torch.float16 else t.contiguous() for t in inputs]
    for name, t in zip(self.input_names, inputs):
      self.context.set_input_shape(name, tuple(t.shape))
      self.context.set_tensor_address(name, t.data_ptr())

    outputs = []
    for name in self.output_names:
      shape = tuple(self.context.get_tensor_shape(name))
      out = torch.empty(shape, dtype=torch.float16, device='cuda')
      self.context.set_tensor_address(name, out.data_ptr())
      outputs.append(out)

    # Enqueue on the current stream and return without synchronizing -- same
    # as any other async CUDA op (a conv, a matmul, ...). Everything in this
    # file and its callers runs on this one stream, so CUDA's own in-order
    # execution guarantees downstream ops (another engine call, F.unfold,
    # ...) won't read these outputs before this one finishes; there is
    # nothing here that requires a CPU-blocking wait. Callers that need the
    # result on the host (e.g. smoke_test's .item()) get a synchronize for
    # free from that host transfer. Do not add a blocking sync here -- it
    # was measured to serialize a per-frame inference loop calling two
    # engines back to back down to ~1/5 the FPS of not blocking.
    stream = torch.cuda.current_stream()
    self.context.execute_async_v3(stream.cuda_stream)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _flatten(out):
  # torch_module(...) returns a single tensor (MainNet) or a tuple (LTE's
  # lv1/lv2/lv3) -- normalize both to one concatenated [B, -1] float tensor
  # so the diff below is a single, simple comparison either way.
  if isinstance(out, torch.Tensor):
    out = (out,)
  return torch.cat([o.reshape(o.size(0), -1).float() for o in out], dim=1)


def smoke_test(name, torch_module, trt_engine_path, dummy_inputs):
  # torch_module and dummy_inputs are already true fp16 (export_onnx_lte/
  # export_onnx_mainnet called .half() on both) -- no autocast needed here.
  with torch.no_grad():
    torch_out = torch_module(*dummy_inputs)
  torch_flat = _flatten(torch_out)

  trt_module = TRTModule(trt_engine_path)
  trt_out = trt_module(*dummy_inputs)
  trt_flat = _flatten(trt_out)

  diff = (torch_flat - trt_flat).abs()
  print('[smoke test] {}: mean abs diff={:.6f}  max abs diff={:.6f}  (random dummy input, sanity check only -- '
        'this is NOT the fp16/argmax-tie-flip drift discussed in optimizations.md, just "does the engine run and '
        'roughly agree with PyTorch")'.format(name, diff.mean().item(), diff.max().item()))


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  device = 'cuda'
  if not torch.cuda.is_available():
    sys.exit("TensorRT conversion requires a CUDA GPU; none is visible to torch.")

  model = TTSR.TTSR(base_args).to(device)
  load_net = torch.load(MODEL_PATH, map_location=device)
  net_sd = model.state_dict()
  net_sd.update(load_net)
  model.load_state_dict(net_sd)
  model.eval()
  del model.LTE_copy
  print('{} LOADED'.format(MODEL_PATH))

  lr_h, lr_w, ref_h, ref_w, scale = compute_shapes(LR_VIDEO, REF_IMG)
  n_feats = base_args.n_feats
  B = BATCH_SIZE
  print('Building engines for batch_size={}, lr={}x{} (lrsr={}x{}), ref={}x{}'.format(
      B, lr_w, lr_h, lr_w * scale, lr_h * scale, ref_w, ref_h))

  # Real ref/ref_sr tensors, and model.LTE run on them in fp32 -- captured
  # BEFORE export_onnx_lte's model.LTE.half() call below mutates the module's
  # weights in place. This is the fp32 baseline the ref-LTE precision check
  # compares the new fp16 TensorRT engine against.
  ref_t, ref_sr_t = load_ref_tensors(REF_IMG, max_ref_dim=384, scale=scale, device=device)
  with torch.no_grad():
    ref_lv1_fp32, ref_lv2_fp32, ref_lv3_fp32 = model.LTE((ref_t.detach() + 1.) / 2.)
    _, _, refsr_lv3_fp32 = model.LTE((ref_sr_t.detach() + 1.) / 2.)

  shape_tag = '{}x3x{}x{}'.format(B, lr_h, lr_w)
  ref_shape_tag = '1x3x{}x{}'.format(ref_h, ref_w)

  # --- LTE (per-frame/batch lrsr shape) ---
  lte_onnx = os.path.join(OUTPUT_DIR, 'LTE_{}.onnx'.format(shape_tag))
  lte_engine = os.path.join(OUTPUT_DIR, 'LTE_{}.engine'.format(shape_tag))
  print('Exporting LTE to ONNX...')
  lte_dummy = export_onnx_lte(model, B, lr_h * scale, lr_w * scale, lte_onnx, OPSET, input_name='lrsr')
  print('Building LTE TensorRT engine (this can take a couple of minutes)...')
  build_engine(lte_onnx, lte_engine, WORKSPACE_GB)
  print('Saved {}'.format(lte_engine))
  smoke_test('LTE', model.LTE, lte_engine, (lte_dummy,))

  # --- LTE (reference-image shape) ---
  # A second, differently-shaped LTE engine for the twice-per-video
  # reference-image calls (ref_lv1/2/3, refsr_lv3) -- these run at the ref
  # image's own resolution (batch_size always 1), not the per-frame lrsr
  # shape above. convert_tensorrt.py originally skipped this (see
  # export_onnx_lte's old docstring) since it only runs twice total and
  # buys no speed; it exists now so test_video_batched.py can run its
  # reference-image LTE calls through TensorRT too instead of loading
  # TTSR.pt for eager fp32 weights. Whether that's actually safe to ship is
  # exactly what the precision check below measures.
  lte_ref_onnx = os.path.join(OUTPUT_DIR, 'LTE_ref_{}.onnx'.format(ref_shape_tag))
  lte_ref_engine = os.path.join(OUTPUT_DIR, 'LTE_ref_{}.engine'.format(ref_shape_tag))
  print('Exporting reference-image LTE to ONNX...')
  lte_ref_dummy = export_onnx_lte(model, 1, ref_h, ref_w, lte_ref_onnx, OPSET, input_name='ref')
  print('Building reference-image LTE TensorRT engine...')
  build_engine(lte_ref_onnx, lte_ref_engine, WORKSPACE_GB)
  print('Saved {}'.format(lte_ref_engine))
  smoke_test('LTE (ref shape, random input)', model.LTE, lte_ref_engine, (lte_ref_dummy,))

  # Real-data precision check (not the random dummy input smoke_test above):
  # run both new engines on the real ref image and real frame 0, then check
  # whether R_lv3_star_arg -- the hard-argmax patch selection -- actually
  # shifts. See check_ref_lte_precision's own comment for why this matters
  # more here than for the per-frame path.
  lte_engine_trt = TRTModule(lte_engine)
  lte_ref_engine_trt = TRTModule(lte_ref_engine)
  # LTE.forward() itself starts with sub_mean(x) expecting x in [0,1] --
  # every real caller (test_video.py/test_video_batched.py) feeds
  # (t.detach() + 1.) / 2. rather than the raw [-1,1] tensor. Match that here.
  ref_lv1_trt, ref_lv2_trt, ref_lv3_trt = lte_ref_engine_trt((ref_t.detach() + 1.) / 2.)
  _, _, refsr_lv3_trt = lte_ref_engine_trt((ref_sr_t.detach() + 1.) / 2.)
  lrsr_t = load_first_frame_lrsr(LR_VIDEO, lr_h, lr_w, scale, device)
  _, _, lrsr_lv3_real = lte_engine_trt((lrsr_t.detach() + 1.) / 2.)
  check_ref_lte_precision(
      (ref_lv1_fp32, ref_lv2_fp32, ref_lv3_fp32, refsr_lv3_fp32),
      (ref_lv1_trt, ref_lv2_trt, ref_lv3_trt, refsr_lv3_trt),
      lrsr_lv3_real,
      names=('ref_lv1', 'ref_lv2', 'ref_lv3', 'refsr_lv3'))

  # --- MainNet ---
  # Needs a much bigger builder workspace than LTE: Myelin fuses essentially
  # the whole network into one opaque kernel and genuinely needs ~51GB to
  # compile it (see the comment in build_engine). If this raises, the most
  # likely cause on a smaller GPU is WORKSPACE_GB above being too low --
  # check the error for "Need <N>" from TensorRT's own verbose logging
  # (trt.Logger.VERBOSE, set at the top of this file) to see the real number.
  mainnet_onnx = os.path.join(OUTPUT_DIR, 'MainNet_{}.onnx'.format(shape_tag))
  mainnet_engine = os.path.join(OUTPUT_DIR, 'MainNet_{}.engine'.format(shape_tag))
  mainnet_ok = False
  try:
    print('Exporting MainNet to ONNX...')
    mainnet_dummy = export_onnx_mainnet(model, B, lr_h, lr_w, n_feats, mainnet_onnx, OPSET)
    print('Building MainNet TensorRT engine (this can take a couple of minutes)...')
    build_engine(mainnet_onnx, mainnet_engine, WORKSPACE_GB)
    print('Saved {}'.format(mainnet_engine))
    smoke_test('MainNet', model.MainNet, mainnet_engine, mainnet_dummy)
    mainnet_ok = True
  except Exception as e:
    print('\nMainNet engine build FAILED: {}'.format(e))
    print('See the comment above this try block -- most likely WORKSPACE_GB '
          'is too small for this GPU. LTE converted successfully above '
          'regardless of this failure.')

  print('\n{} Both LTE engines (per-frame shape {}x{} and reference shape '
        '{}x{}) are specific to batch_size={} and these exact shapes -- '
        'rebuild if LR_VIDEO/REF_IMG/BATCH_SIZE change.'.format(
            'MainNet engine NOT built (see above).' if not mainnet_ok else 'MainNet engine ready.',
            lr_w, lr_h, ref_w, ref_h, B))


if __name__ == '__main__':
  main()
