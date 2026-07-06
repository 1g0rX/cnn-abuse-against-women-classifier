import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import VGG16
import config


def alexnet_scratch():
    model = Sequential(name='AlexNet_Scratch')
    model.add(layers.Conv2D(32, (11, 11), strides=4, padding='valid',
                            input_shape=config.IMG_SHAPE))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same'))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same'))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1024))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(config.NUM_CLASSES, activation='softmax'))

    return model


def build_vgg16_feature_extractor():
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=config.IMG_SHAPE
    )
    base_model.trainable = False
    return base_model


def extract_features(model, images, batch_size=32):
    features = model.predict(images, batch_size=batch_size, verbose=0)
    return features.reshape(features.shape[0], -1)


def build_classifier(input_dim):
    classifier = Sequential(name='Transfer_Classifier')
    classifier.add(layers.Input(shape=(input_dim,)))
    classifier.add(layers.Dense(512, activation='relu'))
    classifier.add(layers.Dropout(0.5))
    classifier.add(layers.Dense(256, activation='relu'))
    classifier.add(layers.Dropout(0.5))
    classifier.add(layers.Dense(config.NUM_CLASSES, activation='softmax'))
    return classifier
