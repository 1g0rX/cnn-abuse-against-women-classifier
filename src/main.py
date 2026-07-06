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
from alexnet import alexnet_scratch
from evaluate import (
    plot_learning_curves, plot_confusion_matrix,
    save_metrics_txt, evaluate_model
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) disponivel(is): {len(gpus)}")
else:
    print("GPU nao disponivel, usando CPU")

print("Carregando dataset...")
images, labels = load_dataset()
print(f"  Total de imagens: {len(images)}")
print(f"  Shape: {images.shape}")
print(f"  Distribuicao: casal={np.sum(labels==0)}, violencia={np.sum(labels==1)}")

X_train, X_test, y_train, y_test = split_dataset(images, labels)
print(f"  Treino: {len(X_train)} imagens")
print(f"  Teste: {len(X_test)} imagens")

print("\nAplicando data augmentation no conjunto de treino...")
X_train_aug, y_train_aug = augment_training_set(X_train, y_train)
print(f"  Treino apos augmentation: {len(X_train_aug)} imagens")

# ============================================================
# EXPERIMENTO A: AlexNet treinada do zero (From Scratch)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENTO A: AlexNet Treinada do Zero (From Scratch)")
print("=" * 70)

metrics_scratch = []
histories_scratch = []
all_y_pred_scratch = []
all_y_true_scratch = None

for run in range(1, config.NUM_RUNS + 1):
    print(f"\n  Execucao {run}/{config.NUM_RUNS}")
    model = alexnet_scratch()
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.15,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    histories_scratch.append(history)

    result = evaluate_model(model, X_test, y_test, history, 'From_Scratch', run)
    metrics_scratch.append({
        'accuracy': result['accuracy'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1': result['f1'],
    })
    all_y_pred_scratch.append(result['y_pred'])
    all_y_true_scratch = result['y_true']
    print(f"    Acurácia: {result['accuracy']*100:.2f}% | "
          f"Precisão: {result['precision']*100:.2f}% | "
          f"Recall: {result['recall']*100:.2f}% | "
          f"F1: {result['f1']*100:.2f}%")

print("\n  Gerando graficos do experimento From Scratch...")
plot_learning_curves(histories_scratch, 'From Scratch')

y_pred_majority_scratch = np.round(np.mean(all_y_pred_scratch, axis=0)).astype(int)
cm_scratch = confusion_matrix(all_y_true_scratch, y_pred_majority_scratch)
plot_confusion_matrix(cm_scratch, 'From Scratch', config.CLASS_NAMES)

save_metrics_txt('From Scratch', metrics_scratch, cm_scratch, config.CLASS_NAMES)

model.save(os.path.join(config.OUTPUT_DIR, 'modelo_from_scratch.keras'))
print(f"  Modelo salvo: {os.path.join(config.OUTPUT_DIR, 'modelo_from_scratch.keras')}")

print("\n" + "=" * 70)
print("FIM DO EXPERIMENTO FROM SCRATCH")
print("=" * 70)
print("\nExecute 'python transfer_only.py' para o experimento de Transfer Learning.")
print("Execute 'python predict.py <imagem> --from-scratch' para classificar uma imagem.")
