import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import seaborn as sns
import config


def plot_learning_curves(history_list, experiment_name):
    all_acc = []
    all_val_acc = []
    all_loss = []
    all_val_loss = []

    min_epochs = min(len(h.history['accuracy']) for h in history_list)

    for h in history_list:
        all_acc.append(h.history['accuracy'][:min_epochs])
        all_val_acc.append(h.history['val_accuracy'][:min_epochs])
        all_loss.append(h.history['loss'][:min_epochs])
        all_val_loss.append(h.history['val_loss'][:min_epochs])

    mean_acc = np.mean(all_acc, axis=0)
    std_acc = np.std(all_acc, axis=0)
    mean_val_acc = np.mean(all_val_acc, axis=0)
    std_val_acc = np.std(all_val_acc, axis=0)
    mean_loss = np.mean(all_loss, axis=0)
    std_loss = np.std(all_loss, axis=0)
    mean_val_loss = np.mean(all_val_loss, axis=0)
    std_val_loss = np.std(all_val_loss, axis=0)

    epochs = range(1, len(mean_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, mean_acc * 100, 'b-', label='Treinamento', linewidth=2)
    ax1.fill_between(epochs, (mean_acc - std_acc) * 100, (mean_acc + std_acc) * 100,
                     alpha=0.2, color='blue')
    ax1.plot(epochs, mean_val_acc * 100, 'r-', label='Validacao', linewidth=2)
    ax1.fill_between(epochs, (mean_val_acc - std_val_acc) * 100,
                     (mean_val_acc + std_val_acc) * 100, alpha=0.2, color='red')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Acuracia (%)')
    ax1.set_title(f'{experiment_name} - Curva de Acuracia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, mean_loss, 'b-', label='Treinamento', linewidth=2)
    ax2.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                     alpha=0.2, color='blue')
    ax2.plot(epochs, mean_val_loss, 'r-', label='Validacao', linewidth=2)
    ax2.fill_between(epochs, mean_val_loss - std_val_loss,
                     mean_val_loss + std_val_loss, alpha=0.2, color='red')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Perda (Loss)')
    ax2.set_title(f'{experiment_name} - Curva de Perda')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = experiment_name.lower().replace(' ', '_')
    path = os.path.join(config.GRAFICOS_DIR, f'{fname}_learning_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Grafico salvo: {path}')

    return mean_acc, mean_val_acc, mean_loss, mean_val_loss


def plot_confusion_matrix(cm, experiment_name, class_names):
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title(f'{experiment_name} - Matriz de Confusao')
    plt.tight_layout()
    fname = experiment_name.lower().replace(' ', '_')
    path = os.path.join(config.GRAFICOS_DIR, f'{fname}_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Matriz de confusao salva: {path}')


def save_metrics_txt(experiment_name, metrics_list, cm_mean, class_names):
    fname = experiment_name.lower().replace(' ', '_')
    path = os.path.join(config.OUTPUT_DIR, f'{fname}_metricas.txt')
    lines = []
    lines.append('=' * 70)
    lines.append(f'RELATORIO DE METRICAS - {experiment_name}')
    lines.append('=' * 70)
    lines.append('')

    metrics_mean = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    metrics_std = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0]}

    lines.append('Metricas Medias (media ± desvio padrao das {} execucoes):'.format(len(metrics_list)))
    lines.append('-' * 50)
    lines.append(f'  Acuracia (Accuracy):    {metrics_mean["accuracy"]*100:.2f}% ± {metrics_std["accuracy"]*100:.2f}%')
    lines.append(f'  Precisao (Precision):   {metrics_mean["precision"]*100:.2f}% ± {metrics_std["precision"]*100:.2f}%')
    lines.append(f'  Recall (Sensibilidade): {metrics_mean["recall"]*100:.2f}% ± {metrics_std["recall"]*100:.2f}%')
    lines.append(f'  F1-Score:               {metrics_mean["f1"]*100:.2f}% ± {metrics_std["f1"]*100:.2f}%')
    lines.append('')

    lines.append('Matriz de Confusao Media (normalizada):')
    lines.append('-' * 50)
    lines.append(f'                  {"  ".join(f"{c:>15}" for c in class_names)}')
    for i, row in enumerate(cm_mean):
        row_str = '  '.join(f'{v:.4f}' for v in row)
        lines.append(f'  {class_names[i]:<15}  {row_str}')
    lines.append('')
    lines.append(f'Arquitetura: AlexNet')
    lines.append(f'Experimento: {experiment_name}')
    lines.append(f'Dataset: {len(class_names)} classes')
    lines.append(f'Numero de execucoes: {len(metrics_list)}')
    lines.append(f'Epocas por execucao: {config.EPOCHS}')
    lines.append('=' * 70)

    content = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  Metricas salvas: {path}')


def evaluate_model(model, X_test, y_test, history, experiment_name, run_idx=None):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
        'y_true': y_test,
    }
