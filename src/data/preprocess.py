#Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2
import random
from  pathlib import Path

IMG_SIZE_DETECTOR = (256,256,3)
IMG_SIZE_CLASSIFIER = (128,128,3)

YOLO_GRID_SIZE = (8,8)

# ----------------------------
# Classifier dataset functions
#This is for the classifier model, not the detector model
def load_single_classification_image(img_dir, label): # Loading function so that the dataset doesn't have to store all of the images at once
    img_dir = img_dir.numpy().decode("utf-8")
    img = cv2.imread(str(img_dir))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img/255.0).astype(np.float32)

    #Random augmentation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0) # Prevent values from being negative or > 1

    return img, label

def make_custom_classification_dataset(paths, labels, batch_size): #A list of the str paths and a list of the int labels
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda img,label: tf.py_function(
        load_single_classification_image, [img,label], [tf.float32, tf.int32]
    ))
    dataset = dataset.map(lambda img, label: (
        tf.ensure_shape(img, IMG_SIZE_CLASSIFIER),
        tf.ensure_shape(label, ()),
    ))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def make_custom_classification_datasets(crops_dir:str, batch_size:int, train_proportion:float):
    crops_dir = Path(crops_dir)
    digits_tuples = []
    for i in range(6): # Digits 0 to 5
        digit_dir = crops_dir / str(i)
        for img_dir in digit_dir.glob("*.jpg"):
            digits_tuples.append((img_dir, i))
    random.seed(42)
    random.shuffle(digits_tuples)
    num_imgs = len(digits_tuples)

    train_size = int(train_proportion*num_imgs)
    val_size = int((num_imgs-train_size)/2)
    test_size = num_imgs - train_size - val_size

    paths = [str(t[0]) for t in digits_tuples]
    labels = [t[1] for t in digits_tuples]

    train_dataset = make_custom_classification_dataset(paths[:train_size], labels[:train_size], batch_size)
    val_dataset = make_custom_classification_dataset(paths[train_size:train_size+val_size], labels[train_size:train_size+val_size], batch_size)
    test_dataset = make_custom_classification_dataset(paths[train_size+val_size:], labels[train_size+val_size:], batch_size)
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size

def preprocess_for_classifier(img):
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     img = cv2.resize(img, (IMG_SIZE_CLASSIFIER[0], IMG_SIZE_CLASSIFIER[1]))
     img = (img/255.0).astype(np.float32)
     img = np.expand_dims(img, axis=0)
     return img
     

# ----------------------------
# Detection dataset functions

#Used for regular  image preprocessing (in live_demo.py or process_video.py)
def preprocess_for_detector(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE_DETECTOR[0], IMG_SIZE_DETECTOR[1]))
    img = (img/255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0)  # add batch dim: (1, 128, 128, 3)
    return img


# Crops an image randomly but ensures that the bounding boxes stay within frame. Rewrites the label based on the new crop. Then resizes the image to the specified input size
def random_crop_with_boxes(img, label):
    img = img.numpy()
    label = label.numpy()
    img_height, img_width = img.shape[:2]
    
    active_boxes = []
    for row in range(YOLO_GRID_SIZE[0]):
        for col in range(YOLO_GRID_SIZE[1]):
            if label[row, col, 0] == 1:
                x = label[row, col, 1] * img_width
                y = label[row, col, 2] * img_height
                w = label[row, col, 3] * img_width
                h = label[row, col, 4] * img_height
                active_boxes.append((x - w/2, y - h/2, x + w/2, y + h/2))
    
    if not active_boxes: #No hands in the image, pointless to crop
        img = cv2.resize(img, (IMG_SIZE_DETECTOR[0], IMG_SIZE_DETECTOR[1]))
        return img.astype(np.float32), label.astype(np.float32)
    
    min_x1 = max(0.0, min(b[0] for b in active_boxes))
    min_y1 = max(0.0, min(b[1] for b in active_boxes))
    max_x2 = min(float(img_width), max(b[2] for b in active_boxes))
    max_y2 = min(float(img_height), max(b[3] for b in active_boxes))
    
    crop_x1 = int(np.random.uniform(0, min_x1)) if min_x1 > 0 else 0
    crop_y1 = int(np.random.uniform(0, min_y1)) if min_y1 > 0 else 0
    crop_x2 = int(np.random.uniform(max_x2, img_width)) if max_x2 < img_width else img_width
    crop_y2 = int(np.random.uniform(max_y2, img_height)) if max_y2 < img_height else img_height
    
    original_ratio = img_width / img_height

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    # Clamp aspect ratio to within 30% of original
    crop_ratio = crop_width / crop_height
    if crop_ratio > original_ratio * 1.3:
        # Too wide — expand height
        target_height = int(crop_width / (original_ratio * 1.3))
        extra = target_height - crop_height
        add_top = extra // 2
        add_bottom = extra - add_top
        crop_y1 = max(0, crop_y1 - add_top)
        crop_y2 = min(img_height, crop_y2 + add_bottom)

    elif crop_ratio < original_ratio * 0.7:
        # Too tall — expand width
        target_width = int(crop_height * original_ratio * 0.7)
        extra = target_width - crop_width
        add_left = extra // 2
        add_right = extra - add_left
        crop_x1 = max(0, crop_x1 - add_left)
        crop_x2 = min(img_width, crop_x2 + add_right)
    
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    img = cv2.resize(img, (IMG_SIZE_DETECTOR[0], IMG_SIZE_DETECTOR[1]))
    
    new_label = np.zeros_like(label)
    for row in range(YOLO_GRID_SIZE[0]):
        for col in range(YOLO_GRID_SIZE[1]):
            if label[row, col, 0] == 1:
                x = label[row, col, 1] * img_width
                y = label[row, col, 2] * img_height
                w = label[row, col, 3] * img_width
                h = label[row, col, 4] * img_height
                
                new_x = (x - crop_x1) / crop_width
                new_y = (y - crop_y1) / crop_height
                new_w = w / crop_width
                new_h = h / crop_height
                
                new_grid_x = np.clip(int(new_x * YOLO_GRID_SIZE[1]), 0, YOLO_GRID_SIZE[1]-1)
                new_grid_y = np.clip(int(new_y * YOLO_GRID_SIZE[0]), 0, YOLO_GRID_SIZE[0]-1)
                
                new_label[new_grid_y, new_grid_x] = [1, new_x, new_y, new_w, new_h, *label[row, col, 5:]]
    return img.astype(np.float32), new_label.astype(np.float32)



def random_flip_with_boxes(img, label):
    img = img.numpy()
    label = label.numpy()
    
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        new_label = np.zeros_like(label)
        for row in range(YOLO_GRID_SIZE[0]):
            for col in range(YOLO_GRID_SIZE[1]):
                if label[row, col, 0] == 1:
                    x = label[row, col, 1]
                    new_x = 1.0 - x
                    new_col = int(new_x * YOLO_GRID_SIZE[1])
                    new_col = np.clip(new_col, 0, YOLO_GRID_SIZE[1]-1)
                    new_label[row, new_col] = label[row, col].copy()
                    new_label[row, new_col, 1] = new_x
        label = new_label
    
    return img.astype(np.float32), label.astype(np.float32)

def parse_detection_example(example_proto, augment=False):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    img = tf.io.decode_jpeg(parsed['image'], channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    label = tf.io.decode_raw(parsed['label'], tf.float32)
    label = tf.reshape(label, (8, 8, 11))
    
    if augment:

        # Random crop first
        img, label = tf.py_function(
            random_crop_with_boxes,
            [img, label],
            [tf.float32, tf.float32]
        )
        img = tf.ensure_shape(img, (256, 256, 3))
        label = tf.ensure_shape(label, (8, 8, 11))

        # Horizontal flip
        img, label = tf.py_function(
            random_flip_with_boxes,
            [img, label],
            [tf.float32, tf.float32]
        )
        img = tf.ensure_shape(img, (256, 256, 3))
        label = tf.ensure_shape(label, (8, 8, 11))
        
        # Brightness and contrast
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
    else:
        img = tf.image.resize(img, [IMG_SIZE_DETECTOR[0], IMG_SIZE_DETECTOR[1]])
        img = tf.ensure_shape(img, (256, 256, 3))
        label = tf.ensure_shape(label, (8, 8, 11))
        
        
    
    return img, label

def parse_train(x):
    return parse_detection_example(x, augment=True)

def parse_val(x):
    return parse_detection_example(x, augment=False)

def make_tf_records_detection_datasets(tfrecord_path, batch_size, train_proportion):
    num_imgs = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    dataset = tf.data.TFRecordDataset(tfrecord_path) # Take the dataset out of a tfrecord file because the images too large to process on the fly
    dataset = dataset.shuffle(num_imgs, seed=42, reshuffle_each_iteration=False)
    

    train_size = int(train_proportion*num_imgs)
    val_size = int((num_imgs-train_size)/2)
    test_size = num_imgs-train_size-val_size
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size+val_size)

    train_dataset = train_dataset.map(parse_train)
    val_dataset = val_dataset.map(parse_val) #Parse with no augmentation
    test_dataset = test_dataset.map(parse_val) 

    train_dataset = train_dataset.repeat().batch(batch_size)
    val_dataset = val_dataset.repeat().batch(batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size
    
    
