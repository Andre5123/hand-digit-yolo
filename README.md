# Hand Digit Recognition with YOLO

Detect and count fingers (0–5) on one or two hands in real time using a webcam, via a custom YOLO-style object detection model built with TensorFlow and OpenCV.

## Demo
![Demo](assets/annotated_recording-v5-sped-up.gif)

## Overview

This project started as a way to put into practice the theory from the DeepLearning.AI CNN course — specifically object detection, custom loss functions, and building an end-to-end ML pipeline from scratch.

The model detects hands in a webcam frame and simultaneously classifies how many fingers are being held up. It supports two hands at once. The whole pipeline — data collection, annotation, training, and inference — was built from scratch.

**Key requirements:** Python 3.12, TensorFlow 2.21, OpenCV, NumPy

## Contents

- **src/live_demo.py** - real-time webcam inference with bounding boxes
- **src/record.py** - record raw webcam footage
- **src/process_video.py** - annotate a recorded video offline
- **src/data/capture.py** - rapid frame capture for data collection
- **src/data/annotate.py** - draw bounding boxes on captured frames
- **src/data/label.py** - assign digit class labels to hand crops
- **src/data/create_tfrecords.py** - convert dataset to TFRecord format
- **src/models/detector.py** - MobileNetV2-based YOLO detector
- **src/models/classifier.py** - standalone VGG-style classifier (used for prototyping)
- **src/training/loss.py** - custom YOLO loss, IoU metric, class accuracy
- **src/training/train_detector.py** - training script for the unified detector

## Usage

Run the live demo:

```bash
cd src
python live_demo.py
```

Press `q` to quit. The model runs detection every few frames using a background thread to keep the display smooth.

To record and annotate a video offline:

```bash
python record.py         # saves raw_recording.mp4 to assets/
python process_video.py  # outputs annotated_recording.mp4
```

## Dataset

All training data was self-collected using the three-script pipeline in `src/data/`. Frames were captured with `capture.py`, bounding boxes were drawn with `annotate.py`, and digit labels were assigned with `label.py`.

The dataset consists of 2,111 annotated images collected across 5 different indoor and outdoor locations, with varied backgrounds, lighting conditions, and hand distances from the camera. 251 images contain no hands at all (negative examples) to reduce false positives. After annotation and labelling, 6,687 labelled crop images were produced for classifier training.

## Model Architecture

The final model is a unified YOLO-style detector that handles both detection and classification in a single forward pass.

**Backbone:** MobileNetV2 pretrained on ImageNet, with the last 50 layers fine-tuned on the custom hand dataset. For a 256×256 input, MobileNetV2 naturally produces an 8×8 feature map which serves as the detection grid.

**Detection head:** A 1×1 Conv2D layer projects from 1280 channels down to 11 — one prediction per grid cell:

```
[objectness, x, y, w, h, p0, p1, p2, p3, p4, p5]
```

where `p0–p5` are class probabilities for digits 0–5.

**Loss function:**

```
L = λ_coord × box_loss + objectness_loss + λ_class × class_loss
```

Box loss is MSE over predicted coordinates, computed only for cells containing a hand. Objectness uses binary cross-entropy with negative cells down-weighted by 0.5 (following the original YOLO paper). Classification uses categorical cross-entropy, also restricted to positive cells.

## Training

Training images are stored in TFRecord format for fast loading. Augmentation is applied on-the-fly during training:

- **Random crops** that preserve all bounding boxes within the frame, simulating different distances and hand positions
- **Horizontal flips** with correct grid cell reassignment for flipped box coordinates  
- **Brightness, contrast, and hue** variation for lighting robustness

A standalone VGG-style classifier was trained first as a baseline before moving to the unified model. It reached 87% validation accuracy on 6,687 labelled crop images. The unified YOLO detector reached 67% validation IoU.

## What Didn't Work

One early attempt used a public ASL digit dataset — the model reached 100% validation accuracy but completely failed on webcam footage. The training images had a consistent dark background, fixed hand position, and no variation in lighting, so the model learned background artifacts rather than hand shapes. Switching to self-collected data that matched the deployment environment fixed this entirely.

A binary edge detection preprocessing step was also explored (adaptive thresholding + morphological operations) before feeding images to the detector, but colour images consistently produced better results.

## Known Limitations

- Performance degrades in very low lighting or with busy backgrounds
- Two hands in the same 8×8 grid cell will only produce one detection (rare in practice)
- Model was trained on one person's hands and may generalise less well to others

## Future Work

- Collect data from multiple contributors for better generalisation
- Export to TensorFlow Lite for mobile deployment
- Add temporal smoothing for more stable bounding boxes
- Support dynamic gestures beyond static digit counts

---

Built by Andre — Computer Engineering, University of Waterloo (incoming 1A)