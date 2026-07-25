import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import tensorflow as tf
import math
def yolo_loss(y_true, y_pred):
    objectness_labels = y_true[..., 0]
    box_coord_labels = y_true[..., 1:5]
    class_labels = y_true[..., 5:]

    objectness_pred = y_pred[..., 0]
    box_coord_pred = y_pred[..., 1:5]
    class_pred = y_pred[..., 5:]

    objectness_mask = (objectness_labels == 1)
    objectness_mask_float = tf.cast(objectness_mask, tf.float32)
    objectness_mask_expanded = objectness_mask_float[..., tf.newaxis]

    # Box loss — only over positive cells
    box_mse = (box_coord_labels - box_coord_pred) ** 2
    box_mse = box_mse * objectness_mask_expanded
    num_positive = tf.reduce_sum(objectness_mask_float) + 1e-7
    box_loss = tf.reduce_sum(box_mse) / num_positive

    # Objectness loss — over all cells, weighted
    objectness_loss = tf.keras.losses.binary_crossentropy(
        objectness_labels[..., tf.newaxis],
        objectness_pred[..., tf.newaxis],
        from_logits=True
    )
    obj_loss_positive = objectness_mask_float * objectness_loss
    obj_loss_negative = (1 - objectness_mask_float) * objectness_loss * 0.5
    objectness_loss_total = tf.reduce_mean(obj_loss_positive + obj_loss_negative)

    # Classification loss — only over positive cells
    class_loss = tf.keras.losses.categorical_crossentropy(
        class_labels,
        class_pred,
        from_logits=True
    )
    class_loss = class_loss * objectness_mask_float
    class_loss_total = tf.reduce_sum(class_loss) / num_positive

    lambda_coord = 5.0
    lambda_class = 1.0

    return lambda_coord * box_loss + objectness_loss_total + lambda_class * class_loss_total

def iou_metric(y_true, y_pred):
    objectness_mask = y_true[...,0]

    true_boxes = y_true[...,1:5]
    pred_boxes = y_pred[...,1:5]
    true_x1 = true_boxes[..., 0] - true_boxes[..., 2]/2
    true_x2 = true_boxes[..., 0] + true_boxes[..., 2]/2
    true_y1 = true_boxes[..., 1] - true_boxes[..., 3]/2
    true_y2 = true_boxes[..., 1] + true_boxes[..., 3]/2

    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2]/2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2]/2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3]/2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3]/2

    inter_x1 = tf.maximum(true_x1, pred_x1)
    inter_x2 = tf.minimum(true_x2, pred_x2)
    inter_y1 = tf.maximum(true_y1, pred_y1)
    inter_y2 = tf.minimum(true_y2, pred_y2)

    inter_width = tf.maximum(0.0, inter_x2-inter_x1)
    inter_height = tf.maximum(0.0, inter_y2-inter_y1)
    intersection = inter_width*inter_height

    true_area = (true_x2-true_x1)*(true_y2-true_y1)
    pred_area = (pred_x2-pred_x1)*(pred_y2-pred_y1)

    union = true_area + pred_area - intersection

    iou = intersection / (union+1e-7)
    iou = iou*objectness_mask

    num_positive = tf.reduce_sum(objectness_mask)+1e-7
    return tf.reduce_sum(iou)/num_positive

def class_accuracy(y_true, y_pred):
    objectness_mask = y_true[..., 0]
    objectness_mask_float = tf.cast(objectness_mask == 1, tf.float32)
    
    true_classes = tf.argmax(y_true[..., 5:], axis=-1)
    pred_classes = tf.argmax(y_pred[..., 5:], axis=-1)
    
    correct = tf.cast(true_classes == pred_classes, tf.float32)
    correct = correct * objectness_mask_float
    
    num_positive = tf.reduce_sum(objectness_mask_float) + 1e-7
    return tf.reduce_sum(correct) / num_positive