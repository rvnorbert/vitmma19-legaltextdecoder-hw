import os
import warnings
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, TFDistilBertModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import config
from utils import setup_logger

warnings.filterwarnings("ignore")

logger = setup_logger()

def evaluate():
    if not os.path.exists(config.TEST_FILE):
        logger.error("Test file not found.")
        return
    if not os.path.exists(config.MODEL_FILE_KERAS):
        logger.error(f"Model file not found: {config.MODEL_FILE_KERAS}")
        return

    logger.info("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_DIR)
    model = tf.keras.models.load_model(
        config.MODEL_FILE_KERAS,
        custom_objects={'TFDistilBertModel': TFDistilBertModel}
    )

    df = pd.read_csv(config.TEST_FILE)
    texts = df['text'].tolist()
    y_true = df['label'].tolist()

    logger.info(f"Predicting on test set ({len(texts)} samples)...")
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=config.MAX_LEN, return_tensors='tf')

    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}
    )).batch(config.BATCH_SIZE)

    preds = model.predict(dataset, verbose=0)
    probs = tf.nn.softmax(preds, axis=-1).numpy()
    y_pred = np.argmax(probs, axis=1)

    y_true_orig = [y + 1 for y in y_true]
    y_pred_orig = [y + 1 for y in y_pred]

    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")

    report = classification_report(y_true_orig, y_pred_orig, zero_division=0)
    logger.info("\n" + report)

    with open(config.EVAL_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true_orig, y_pred_orig, labels=[1, 2, 3, 4, 5])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.gca().invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Legal Text Decoder - Confusion Matrix')
    plt.savefig(config.CONFUSION_MATRIX_FILE)
    logger.info(f"Confusion matrix saved to: {config.CONFUSION_MATRIX_FILE}")

if __name__ == "__main__":
    evaluate()