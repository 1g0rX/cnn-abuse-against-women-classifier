import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

import config
from preprocess import load_dataset, split_dataset, augment_training_set
from alexnet import build_vgg16_feature_extractor, extract_features, build_classifier
from evaluate import (
    plot_learning_curves, plot_confusion_matrix,
    save_metrics_txt, evaluate_model
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("Carregando dataset...")
images, labels = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(images, labels)
X_train_aug, y_train_aug = augment_training_set(X_train, y_train)

print("Extraindo features com VGG16 (pre-treinado ImageNet)...")
vgg16_base = build_vgg16_feature_extractor()
X_train_features = extract_features(vgg16_base, X_train_aug)
X_test_features = extract_features(vgg16_base, X_test)
print(f"  Features shape (treino): {X_train_features.shape}")
print(f"  Features shape (teste): {X_test_features.shape}")

input_dim = X_train_features.shape[1]

NUM_RUNS = 3
EPOCHS = 30

print("\n" + "=" * 70)
print("EXPERIMENTO B: Transfer Learning (VGG16 + Classificador Treinado)")
print("=" * 70)

metrics_transfer = []
histories_transfer = []
all_y_pred_transfer = []
all_y_true_transfer = None

for run in range(1, NUM_RUNS + 1):
    print(f"\n  Execucao {run}/{NUM_RUNS}")

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
    )

    classifier = build_classifier(input_dim)
    classifier.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = classifier.fit(
        X_train_features, y_train_aug,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    histories_transfer.append(history)

    y_pred_prob = classifier.predict(X_test_features, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    metrics_transfer.append({
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
    })
    all_y_pred_transfer.append(y_pred)
    all_y_true_transfer = y_test
    print(f"    Acurácia: {acc*100:.2f}% | "
          f"Precisão: {prec*100:.2f}% | "
          f"Recall: {rec*100:.2f}% | "
          f"F1: {f1*100:.2f}%")

print("\n  Gerando graficos do experimento Transfer Learning...")
plot_learning_curves(histories_transfer, 'Transfer Learning')

y_pred_majority = np.round(np.mean(all_y_pred_transfer, axis=0)).astype(int)
cm_transfer = confusion_matrix(all_y_true_transfer, y_pred_majority)
plot_confusion_matrix(cm_transfer, 'Transfer Learning', config.CLASS_NAMES)

save_metrics_txt('Transfer Learning', metrics_transfer, cm_transfer, config.CLASS_NAMES)

classifier.save(os.path.join(config.OUTPUT_DIR, 'classificador_transfer.keras'))
vgg16_base.save(os.path.join(config.OUTPUT_DIR, 'extrator_vgg16.keras'))
print(f"  Classificador salvo: {os.path.join(config.OUTPUT_DIR, 'classificador_transfer.keras')}")
print(f"  Extrator VGG16 salvo: {os.path.join(config.OUTPUT_DIR, 'extrator_vgg16.keras')}")

print("\nRELATORIO COMPARATIVO FINAL")
print("=" * 70)
mean_acc_t = np.mean([m['accuracy'] for m in metrics_transfer]) * 100
std_acc_t = np.std([m['accuracy'] for m in metrics_transfer]) * 100
print(f"\nTransfer Learning - Acuracia media: {mean_acc_t:.2f}% ± {std_acc_t:.2f}%")
print(f"\nResultados salvos em: {config.OUTPUT_DIR}")
