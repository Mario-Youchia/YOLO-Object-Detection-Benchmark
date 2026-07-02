# YOLO Object Detection Model Comparison

This project benchmarks multiple pretrained YOLO object-detection models on the same traffic video. It compares total frame-level detections and model-reported processing time on CPU and GPU, while saving one annotated output video for each unique model.

![Model output comparison](assets/model-output-comparison.png)

## Models

The consolidated implementation uses these unique Ultralytics model weights:

- YOLOv3u
- YOLOv5n-u
- YOLOv5s-u
- YOLOv5m-u
- YOLOv5l-u
- YOLOv5x-u
- YOLOv8n

The original notebook also called `yolov5m.pt`. Ultralytics redirected that legacy filename to `yolov5mu.pt`, so the resulting video was byte-for-byte identical to the YOLOv5m-u output. The duplicate video and duplicate CPU/GPU copies were removed from this package.

## Original benchmark results

These values were preserved from the executed project notebook and its report. Processing time is the accumulated `preprocess + inference + postprocess` time reported by Ultralytics for each readable frame; it is not full wall-clock runtime.

| Model run | Total frame detections | CPU time | GPU time |
|---|---:|---:|---:|
| YOLOv3u | 2,614 | 596 s | 7 s |
| YOLOv5m-u | 2,583 | 166 s | 3 s |
| YOLOv8n | 2,483 | 28 s | 1 s |
| YOLOv5n, legacy name | 2,272 | 25 s | 1 s |
| YOLOv5s, legacy name | 2,428 | 62 s | 2 s |
| YOLOv5m, legacy alias | 2,583 | 156 s | 2 s |
| YOLOv5l, legacy name | 2,668 | 298 s | 4 s |
| YOLOv5x, legacy name | 2,719 | 530 s | 6 s |

![Runtime comparison](assets/runtime-comparison.png)

The count represents detections summed across video frames, not unique tracked physical objects.

## Repository structure

```text
.
├── README.md
├── yolo_object_detection_benchmark.ipynb
├── requirements.txt
├── data/
│   └── Test_sample_video.mp4
├── outputs/
│   ├── yolov3u.mp4
│   ├── yolov5nu.mp4
│   ├── yolov5su.mp4
│   ├── yolov5mu.mp4
│   ├── yolov5lu.mp4
│   ├── yolov5xu.mp4
│   └── yolov8n.mp4
├── results/
│   └── original_benchmark_results.csv
└── assets/
    ├── model-output-comparison.png
    └── runtime-comparison.png
```

Pretrained `.pt` weight files are intentionally not included. Ultralytics downloads them when the notebook is executed, avoiding hundreds of megabytes of duplicated third-party model files.

## Run on Kaggle

1. Upload this project folder or ZIP as a Kaggle Dataset.
2. Create a Kaggle Notebook and attach the dataset.
3. Enable a GPU accelerator when GPU benchmarking is required.
4. Upload/open `yolo_object_detection_benchmark.ipynb`.
5. Run the cells in order.

The notebook automatically searches `/kaggle/input` for `Test_sample_video.mp4` and writes new files under `/kaggle/working/yolo-benchmark`.

## Run locally

```bash
python -m pip install -r requirements.txt
jupyter notebook yolo_object_detection_benchmark.ipynb
```

## Methodology notes

- Every model processes the same readable video frames.
- CPU and GPU execution are selected explicitly through the Ultralytics `device` argument.
- Annotated videos are written using the source video's dimensions and frame rate.
- Only one visual output is saved per unique model; CPU and GPU output copies are not duplicated.
- The notebook records both Ultralytics model time and end-to-end wall-clock time for future reruns.

## Known limitation

The provided MP4 advertises roughly 1,745 frames in its container metadata, but decoding stops after 145 readable frames. The included historical outputs and measurements therefore cover those 145 readable frames, producing approximately 5.8 seconds of video at 25 FPS. This source-file limitation is detected and reported by the consolidated notebook.

Rerunning with newer Ultralytics, PyTorch, CUDA, or hardware versions can produce different timings and occasionally different detections. The table above should be treated as the historical result of the original run, while newly generated CSV files are the result of the current environment.
