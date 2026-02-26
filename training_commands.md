Rhythm Quantizer:
torchrun --nproc_per_node=4 --master_port=29501 -m training.train_pm2s \
  --task rhythm_quantizer \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --wandb \
  --wandb-name "pm2s_{task}_test" \
  --gpu_ids 4,5,6,7 \
  --batch-size 4 \
  --grad-accum-steps 32 \
  --amp \
  --epochs 20 \
  --amp-dtype bf16 \
  --time-scale 100 \
  --tempo-loss log \
  --tempo-loss-weight 5.0 \
  --beat-num-loss-weight 0.1 \
  --warmup-steps 1000 \
  --min-lr-ratio 0.01 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --sample-by-dataset-weights \
  --eval-best-swd09 \
  --save-every 5

Voice Separator:
torchrun --nproc_per_node=2 --master_port=29501 -m training.train_pm2s \
  --task voice_separator \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --wandb \
  --wandb-name "pm2s_{task}_test" \
  --gpu_ids 6,7 \
  --batch-size 4 \
  --grad-accum-steps 64 \
  --amp \
  --epochs 20 \
  --amp-dtype bf16 \
  --time-scale 100 \
  --warmup-steps 50 \
  --sample-by-dataset-weights \
  --save-every 5

Key Estimator:
torchrun --nproc_per_node=4 --master_port=29501 -m training.train_pm2s \
  --task key_estimator \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --wandb \
  --wandb-name "pm2s_{task}_test" \
  --gpu_ids 4,5,6,7 \
  --batch-size 4 \
  --grad-accum-steps 64 \
  --amp \
  --epochs 20 \
  --amp-dtype bf16 \
  --time-scale 100 \
  --warmup-steps 30 \
  --sample-by-dataset-weights \
  --save-every 5

Notes:
- Keep `--warmup-steps` small for short test runs (~5–10% of total optimizer steps).
- `--sample-by-dataset-weights` is important because PDMX dominates the manifests; it requires a per-item `source` (or `datasets/pm2s_manifests/manifest_state.json` next to the manifests).
- If SWD eval stays low, check whether RhythmQuantizer is learning tempo/time signature with `python evaluation/debug_rhythm_quantizer_swd.py --ckpt checkpoints/pm2s/rhythm_quantizer/epoch_5.pt`.


"Oh, finally we got some results! Averages
  baseline: onset_f1=0.8220, onset+offset_f1=0.7819, frame_f1=0.9090, time=883.0 ms
  pm2s    : onset_f1=0.4782, onset+offset_f1=0.4637, frame_f1=0.6390, time=667.7 ms

Please update CODEX.md with everything that you understood and fixed, so that we won't lose our progress. I checked /workspace/huseyn/audio-to-score/outputs/Schubert_D911-21.pm2s.musicxml manually and noticed that there are A LOT AND BIG gaps between notes, and that's why it's 6:56 minutes long (original is 2:04). Present ideas to fix this."
