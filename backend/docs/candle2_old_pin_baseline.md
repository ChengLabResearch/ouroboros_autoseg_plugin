# CANDLE-2 old-pin Medical-SAM3 baseline

Recorded 2026-07-18 for `den-sq/sam_parity#34` and the synchronized comparison
required by `den-sq/sam_parity#40`.

## Fixed inputs

- Plugin: `c8d3c0218acde2c31f77f9a37d630368c83667cc`
- Candle: `770d20ca8db4f834ba4c89c845bca196fbfc97ea`
- CUDA image: `sha256:6361ad063e2f0d44757d53b2fc3284bb3395a8bc2f9c44c09199d795a4d0c12f`
- GPU: Quadro RTX 5000 with Max-Q Design, compute capability 7.5
- Driver / CUDA runtime: 581.60 / 12.4.1
- Medical checkpoint: `checkpoint_3D.pt`, SHA-256 `6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6`
- Fixture: first 16 pages of `psd_anno_thing.tif`, 16x200x200 `uint16`, with the frame-0 annotation retained as one tracked object
- Fixture SHA-256: `48dd32e122e73582a0db13207384f34faa1578098e7e8578f2876084cea06796`

## Results

| Run | Elapsed | Throughput | Peak VRAM | Average GPU utilization | Peak container RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 92 s | 0.174 fps | 11,936 MiB | 70.8% | 333.7 MiB |
| Warm | 98 s | 0.163 fps | 11,927 MiB | 64.2% | 325.7 MiB |

Both runs passed the harness checks for CUDA device 0, exact plugin/Candle SHAs,
16 output pages at 200x200, `uint8` output, and values restricted to 0 and 255.
The cold and warm TIFF outputs were byte-identical with SHA-256
`bf6d366d33adc09053ba6da7065c88916d6780a133bb2779204f458af751c67e`.

The API remained at its static 30% inference marker during propagation. The
separate frame-derived progress work remains in `den-sq/sam_parity#38`.

## Reproduction

Build the fixed CUDA control:

```bash
docker build -f backend/Dockerfile --target cuda-runtime \
  --build-arg CANDLE_FEATURES=cuda \
  --build-arg CUDA_COMPUTE_CAP=75 \
  --build-arg CANDLE_SAM3_COMMIT=770d20ca8db4f834ba4c89c845bca196fbfc97ea \
  --build-arg PLUGIN_GIT_SHA=c8d3c0218acde2c31f77f9a37d630368c83667cc \
  -t ouroboros-autoseg-backend:candle2-baseline backend
```

Then run `backend/scripts/biological_video_smoke_gpu.sh` with
`INPUT_STACK`, `CHECKPOINT_PATH`, and `ARTIFACT_DIR` set. The script records
`revisions.env`, synchronized `telemetry.csv`, a bounded `backend.log`, and the
validated output TIFF. Use the same fixture and settings with only
`CANDLE_SAM3_COMMIT` changed for the CANDLE-2.2 comparison.
