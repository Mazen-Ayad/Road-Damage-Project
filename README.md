# Road Damage Detection with YOLO

Detect and localize road damage (7 classes) in photos using YOLO object detection, compare three YOLO versions, and deploy the best one as a Gradio app.

**Dataset:** [road-damage](https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7) by TrafficSignsSafeway on Roboflow (CC BY 4.0), 6,768 images, 7 classes: `alligator, block, crack, edge, longitudinal, pothole, transverse`.
Split used: 4,930 train / 1,146 validation / 681 test images (5 train and 4 validation images were ignored by Ultralytics because their label files mix polygon and box rows).

## What was done

- Trained **YOLOv8s, YOLOv10s and YOLO11s** (COCO-pretrained, Ultralytics 8.4) on a Kaggle Tesla T4.
- Hyperparameter tuning, round 1: two optimizer setups per model, **AdamW (lr0 = 0.001)** vs **SGD (lr0 = 0.01)**, giving 6 runs. Everything else was kept identical: 40 epochs, image size 640, batch 16, early-stopping patience 10, seed 0.
- Hyperparameter tuning, round 2 (on the best model, YOLOv10s + SGD, Google Colab T4): **80 epochs**, **image size 800** and **lr0 = 0.005**, one change at a time.
- Compared all runs on the held-out **test set** (mAP, precision, recall, IoU, speed, model size).
- Deployed the final model (**YOLOv10s + SGD, lr0 = 0.005**) as a Gradio app (`app.py`).

## Test-set results

| Run | mAP50 | mAP50-95 | Precision | Recall | Mean IoU* | Size (MB) | Params (M) | GPU FPS** | CPU FPS*** |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv10s + SGD** | **0.612** | **0.384** | 0.597 | 0.595 | 0.778 | **16.5** | 8.1 | 96.7 | 6.7 |
| YOLO11s + SGD | 0.600 | 0.374 | 0.587 | 0.591 | 0.779 | 19.2 | 9.4 | 93.0 | 6.5 |
| YOLOv8s + SGD | 0.592 | 0.365 | 0.590 | 0.578 | 0.775 | 22.5 | 11.1 | 95.9 | 5.7 |
| YOLOv10s + AdamW | 0.600 | 0.363 | 0.615 | 0.565 | 0.777 | 16.5 | 8.1 | 95.7 | 6.6 |
| YOLO11s + AdamW | 0.585 | 0.363 | 0.565 | 0.584 | 0.780 | 19.2 | 9.4 | 89.5 | 6.5 |
| YOLOv8s + AdamW | 0.580 | 0.358 | 0.616 | 0.555 | 0.776 | 22.5 | 11.1 | 88.1 | 5.7 |

\* Mean IoU of detections matched to a ground-truth box of the same class (confidence >= 0.25, IoU >= 0.5), so it is always above 0.5 and does not separate the models well. mAP50-95 is the better localization-quality metric.
\*\* End-to-end (pre-process + inference + post-process), batch size 1, image size 640, Tesla T4. Repeated runs of the same architecture differ by about 5%, so small speed differences are noise.
\*\*\* Batch size 1 on the Kaggle CPU, 100 test images.

Per-class AP50 on the test set (SGD runs):

| Class | YOLOv8s | YOLOv10s | YOLO11s |
|---|---|---|---|
| block | 0.842 | 0.874 | 0.848 |
| crack | 0.830 | 0.854 | 0.839 |
| pothole | 0.695 | 0.697 | 0.696 |
| transverse | 0.600 | 0.628 | 0.618 |
| alligator | 0.345 | 0.358 | 0.344 |
| longitudinal | 0.241 | 0.261 | 0.258 |

The `edge` class has 0 test instances (21 in train, 1 in validation), so it is excluded from the test means and cannot be evaluated.

## Tuning round 2 (YOLOv10s + SGD, test set)

| Run | Image size | Epochs | Val mAP50-95 | mAP50 | **mAP50-95** | Precision | Recall | GPU FPS | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (lr0 0.01) | 640 | 40 | 0.256 | 0.612 | 0.384 | 0.597 | 0.595 | 96.7 (Kaggle) | 16.5 |
| 80 epochs | 640 | 80 | 0.253 | 0.600 | 0.376 | 0.598 | 0.599 | 69.4 (Colab) | 16.5 |
| Image size 800 | 800 | 40 | 0.251 | 0.610 | 0.377 | 0.607 | 0.577 | 46.3 (Colab) | 16.5 |
| **lr0 = 0.005 (final)** | 640 | 40 | **0.259** | **0.621** | **0.390** | 0.600 | 0.621 | 73.4 (Colab) | 16.5 |

FPS values in this table were measured on two different machines (baseline on Kaggle, the others on Colab), so only the Colab rows are comparable with each other.

Per-class AP50 on the test set: block 0.889, crack 0.859, pothole 0.693, transverse 0.657, alligator 0.361, longitudinal 0.268 (final model).

- **Lower learning rate gave the best result** on both the validation set (0.259) and the test set (0.390), but the gain over the baseline is only +0.006 mAP50-95. With a single seed, this is within run-to-run noise, so it is a slight edge and not a proven improvement.
- **Training for 80 epochs did not help** (test 0.376 vs 0.384). The first guess that the 40-epoch models had not converged was not supported: validation mAP50-95 plateaued around 0.25 after roughly epoch 55.
- **Larger images (800) did not help overall** (test 0.377) and cost about 35% of the speed. It gave small gains on the thin classes (longitudinal 0.267 vs 0.261, alligator 0.378 vs 0.358), but not enough to change the picture.

## Analysis

- **The three models are close in accuracy.** The spread in mAP50-95 between the best and worst run is 0.026, and all runs use a single seed, so only large gaps should be read as real. YOLOv10s + SGD is best on both the validation (0.256) and the test set (0.384), and it is also the smallest model, which makes it the best overall trade-off.
- **SGD beat AdamW for all three models on the test set** (+0.007 to +0.021 mAP50-95). The direction is consistent but the size is small, so this is a tendency, not a proven effect.
- **More training and larger images were not the bottleneck.** In round 1 the best epoch was 38-40 and early stopping never triggered, which suggested longer training might help; round 2 showed it did not (80 epochs: 0.376 vs 0.384). The limit is more likely the data (thin, similar-looking classes and inconsistent labels) than the training budget.
- **The class matters more than the model.** Blocks and cracks are detected well; longitudinal cracks and alligator cracking are weak in every model. They are thin, irregular, visually similar to each other, and their labels vary.
- **Class imbalance:** pothole (6,075 train boxes), alligator (4,337) and longitudinal (2,008) dominate, while block has 305 and edge 21.
- **Split caveat:** crack AP50 is about 0.26-0.29 on the validation set (223 boxes) but about 0.83 on the test set (93 boxes). With so few boxes, and class proportions that differ between splits, per-class numbers for the small classes are unstable and depend on the split.
- **Deployment:** all models run at roughly 90-100 FPS end-to-end on a T4 GPU and about 6 FPS on a CPU. YOLOv10s is the smallest (16.5 MB) and is NMS-free, so it is the easiest to ship. For a CPU-only host, expect a few frames per second, which is fine for image-based use but not for live video.

## Limitations

- Limited tuning (optimizer, learning rate, epochs and image size, one change at a time, no combined search), single seed. Differences below about 0.01 mAP50-95 should not be treated as real.
- The `edge` class is effectively absent from the data.
- The test set is small (681 images, 2,004 boxes).

## Run the app

```bash
pip install -r requirements.txt
python app.py            # opens at http://127.0.0.1:7860
python app.py --share    # also prints a temporary public link (works while the app is running)
```

Put the trained weights next to `app.py` as `best.pt` (or set the `MODEL_PATH` environment variable). The final model is the file `v10s_sgd_lr0005.pt` from the tuning round; rename it to `best.pt`. Hugging Face now requires a paid plan for Gradio Spaces, so the app is run locally (or through the temporary `--share` link).

## Credits

Dataset: road-damage by TrafficSignsSafeway on Roboflow, CC BY 4.0. Models: Ultralytics YOLO.
