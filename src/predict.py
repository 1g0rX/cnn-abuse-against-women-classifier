import os
import sys
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from alexnet import build_vgg16_feature_extractor

IMG_SIZE = 224


def preprocess_single_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Nao foi possivel ler a imagem: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float32) / 255.0
    return img


def predict_from_scratch(img_path, modelo_path):
    from tensorflow.keras.models import load_model
    img = preprocess_single_image(img_path)
    model = load_model(modelo_path)
    prob = model.predict(img.reshape(1, IMG_SIZE, IMG_SIZE, 3), verbose=0)[0]
    classe = np.argmax(prob)
    confianca = prob[classe] * 100
    return classe, confianca, prob


def predict_transfer(img_path, extrator_path, classificador_path):
    from tensorflow.keras.models import load_model
    img = preprocess_single_image(img_path)
    extrator = load_model(extrator_path)
    classificador = load_model(classificador_path)
    features = extrator.predict(img.reshape(1, IMG_SIZE, IMG_SIZE, 3), verbose=0)
    features_flat = features.reshape(1, -1)
    prob = classificador.predict(features_flat, verbose=0)[0]
    classe = np.argmax(prob)
    confianca = prob[classe] * 100
    return classe, confianca, prob


def main():
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_da_imagem> [--from-scratch | --transfer]")
        print("Exemplo: python predict.py ~/Downloads/violencia2.jpeg --transfer")
        sys.exit(1)

    img_path = os.path.expanduser(sys.argv[1])
    modo = 'transfer'
    if len(sys.argv) >= 3:
        modo = sys.argv[2].replace('--', '')

    if not os.path.exists(img_path):
        print(f"Erro: arquivo nao encontrado: {img_path}")
        sys.exit(1)

    nomes = ['Nao Violencia (Casal)', 'Violencia']

    if modo == 'from-scratch':
        modelo = os.path.join(config.OUTPUT_DIR, 'modelo_from_scratch.keras')
        if not os.path.exists(modelo):
            print("Erro: execute 'main.py' primeiro para treinar e salvar o modelo From Scratch.")
            sys.exit(1)
        classe, conf, probs = predict_from_scratch(img_path, modelo)
        print(f"\n--- Predicao (From Scratch / AlexNet) ---")
    else:
        extrator = os.path.join(config.OUTPUT_DIR, 'extrator_vgg16.keras')
        classificador = os.path.join(config.OUTPUT_DIR, 'classificador_transfer.keras')
        if not os.path.exists(extrator) or not os.path.exists(classificador):
            print("Erro: execute 'transfer_only.py' primeiro para treinar e salvar os modelos.")
            sys.exit(1)
        classe, conf, probs = predict_transfer(img_path, extrator, classificador)
        print(f"\n--- Predicao (Transfer Learning / VGG16) ---")

    print(f"  Arquivo: {img_path.split('/')[-1]}")
    print(f"  Classe: {nomes[classe]}")
    print(f"  Confianca: {conf:.2f}%")
    print(f"  Probabilidades:")
    print(f"    {nomes[0]}: {probs[0]*100:.2f}%")
    print(f"    {nomes[1]}: {probs[1]*100:.2f}%")


if __name__ == '__main__':
    main()
