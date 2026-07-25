import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('.')

import numpy as np
import tensorflow as tf
import cv2
import threading
from data.preprocess import preprocess_for_detector
from training.loss import yolo_loss, iou_metric, class_accuracy

detector = tf.keras.models.load_model("../models/detector_v13_iou.keras",
    custom_objects={"yolo_loss": yolo_loss, "iou_metric": iou_metric, "class_accuracy": class_accuracy})

cam = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

recording = True

if recording:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('../assets/raw_recording.mp4', fourcc, 20.0, (frame_width, frame_height))

latest_boxes = []
latest_frame = None
frame_lock = threading.Lock()
running = True

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-7)

def nms(boxes, iou_threshold=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if compute_iou(best, b) < iou_threshold]
    return kept

def decode_predictions(output, frame_width, frame_height, threshold=0.3):
    boxes = []
    output = output[0]
    for row in range(8):
        for col in range(8):
            objectness = 1 / (1 + np.exp(-output[row, col, 0])) #Sigmoid
            if objectness > threshold:
                x = output[row, col, 1]
                y = output[row, col, 2]
                w = output[row, col, 3]
                h = output[row, col, 4]
                class_id = np.argmax(output[row, col, 5:])
                x1 = int((x - w/2) * frame_width)
                y1 = int((y - h/2) * frame_height)
                x2 = int((x + w/2) * frame_width)
                y2 = int((y + h/2) * frame_height)
                boxes.append((x1, y1, x2, y2, objectness, class_id))
    return nms(boxes)

def detection_thread():
    global latest_boxes
    while running:
        with frame_lock:
            frame = latest_frame
        if frame is not None:
            detector_processed = preprocess_for_detector(frame)
            detector_output = detector(detector_processed, training=False).numpy()
            latest_boxes = decode_predictions(detector_output, frame_width, frame_height)

thread = threading.Thread(target=detection_thread, daemon=True)
thread.start()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    if recording:
        out.write(frame)  # always fast, never blocked

    with frame_lock:
        latest_frame = frame.copy()

    display = frame.copy()
    for box in latest_boxes:
        x1, y1, x2, y2, objectness, class_id = box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"{class_id} ({objectness:.2f})",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Camera', display)

    if cv2.waitKey(1) == ord('q'):
        break

running = False
cam.release()
out.release()
cv2.destroyAllWindows()