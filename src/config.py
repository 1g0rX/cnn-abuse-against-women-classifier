import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'mulheres')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados')
GRAFICOS_DIR = os.path.join(OUTPUT_DIR, 'graficos')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 16
EPOCHS = 25
NUM_RUNS = 3
TEST_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001

CLASSES = ['casal', 'violencia']
CLASS_NAMES = ['Nao Violencia', 'Violencia']
NUM_CLASSES = len(CLASSES)
