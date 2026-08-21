import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-video1', '--video1', type = str, required = True, help = 'path to the first input video')
parser.add_argument('-video2', '--video2', type = str, required = True, help = 'path to the second input video')
parser.add_argument('-alpha', '--alpha', type = float, required = True, help = 'multiplier applied to the per-pixel abs difference')
parser.add_argument('-output', '--output', type = str, required = True, help = 'path to save the diff video')

param = parser.parse_args()

cap1 = cv2.VideoCapture(param.video1)
if not cap1.isOpened():
  raise RuntimeError("Cannot open video: {}".format(param.video1))

cap2 = cv2.VideoCapture(param.video2)
if not cap2.isOpened():
  raise RuntimeError("Cannot open video: {}".format(param.video2))

fps = cap1.get(cv2.CAP_PROP_FPS)
if fps <= 0:
  fps = 30.0

writer = None
frame_count = 0
while True:
  ret1, frame1 = cap1.read()
  ret2, frame2 = cap2.read()
  if not ret1 or not ret2:
    break

  if frame2.shape[:2] != frame1.shape[:2]:
    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_CUBIC)

  diff = cv2.absdiff(frame1, frame2).astype(np.float32)
  diff = np.clip(diff * param.alpha, 0, 255).astype(np.uint8)

  if writer is None:
    h, w = diff.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(param.output, fourcc, fps, (w, h))

  writer.write(diff)
  frame_count += 1

cap1.release()
cap2.release()
if writer is not None:
  writer.release()

if frame_count == 0:
  raise RuntimeError("No frames written -- check that both videos opened correctly")

print('{} frames written to {}'.format(frame_count, param.output))
