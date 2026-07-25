import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # To be able to access files in another folder

import tensorflow as tf

from models.detector import build_detector
from data.preprocess import make_tf_records_detection_datasets
from loss import yolo_loss, iou_metric, class_accuracy

import math

model = build_detector()
'''
model = tf.keras.models.load_model(
    "../../models/detector_v9.keras",
    custom_objects={"yolo_loss": yolo_loss, "iou_metric": iou_metric, "class_accuracy": class_accuracy}
    ) # Toggle to keep improving on existing model
'''
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=yolo_loss,
    metrics=[iou_metric, class_accuracy]
)

model.summary()

train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = make_tf_records_detection_datasets("../../data/custom/detection.tfrecord", 32, 0.7)

print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

model.fit(
    train_dataset,
    epochs=600,
    steps_per_epoch=math.ceil(train_size/32),
    validation_data=val_dataset,
    validation_steps=math.ceil(val_size/32),
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../../models/detector_best.keras', # Save the model with the best loss
            save_best_only=True,
            monitor='val_loss',
            mode='min',
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../../models/detector_best_iou.keras', # Save the model with the best IOU
            save_best_only=True,
            monitor='val_iou_metric',
            mode='max',
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=600,
            monitor='loss',
            mode='min',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=200,
            monitor='loss',
            mode='min',
            min_lr=1e-8
        )
    ]
)
