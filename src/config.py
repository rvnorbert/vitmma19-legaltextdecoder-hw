import os

# Paths
BASE_DIR = '/app'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# File names
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.csv')
VAL_FILE = os.path.join(OUTPUT_DIR, 'val.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.csv')

# Model save paths
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'legal_text_model') # Tokenizer
MODEL_FILE_KERAS = os.path.join(OUTPUT_DIR, 'legal_model_final.keras') # Weights
EVAL_REPORT_FILE = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
CONFUSION_MATRIX_FILE = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')

# Hyperparameters
MODEL_NAME = "distilbert-base-multilingual-cased"
BATCH_SIZE = 8
EPOCHS = 20
MAX_LEN = 128
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# Label settings
NUM_LABELS = 5