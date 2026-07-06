import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import config

def load_dataset():
    images = []
    labels = []
    for idx, cls in enumerate(config.CLASSES):
        cls_dir = os.path.join(config.DATASET_DIR, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(cls_dir, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                img = cv2.GaussianBlur(img, (5, 5), 0)
                images.append(img)
                labels.append(idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    images, labels = shuffle(images, labels, random_state=config.RANDOM_STATE)
    return images, labels


def split_dataset(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_STATE
    )
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test, y_train, y_test


def augment_training_set(X_train, y_train):
    augmented_images = []
    augmented_labels = []
    datagen_config = {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'brightness_range': (0.8, 1.2),
        'zoom_range': 0.1,
    }
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(**datagen_config)
    for img, label in zip(X_train, y_train):
        img_4d = img.reshape((1, config.IMG_SIZE, config.IMG_SIZE, 3))
        aug_iter = datagen.flow(img_4d, batch_size=1)
        augmented_images.append(img)
        augmented_labels.append(label)
        for _ in range(2):
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)
