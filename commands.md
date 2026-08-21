Ananya test
```
python test_video.py   \
--model TTSR.pt  \
--ref_img /workspace/highres_images/ananya.png \
--lr_video /workspace/lowres/ananya.mp4 \
--save_path /workspace/outputs/ananya_sr.mp4
```

Shourya test
```
python test_video.py   \
--model TTSR.pt  \
--ref_img /workspace/highres_images/shourya.png \
--lr_video /workspace/lowres/shourya_bunty_2.mp4 \
--save_path /workspace/outputs/shourya_bunty_sr.mp4
```