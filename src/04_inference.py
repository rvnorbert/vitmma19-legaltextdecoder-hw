import os
import argparse
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertModel
import config
from utils import setup_logger

logger = setup_logger()


def predict_text(text):
    if not os.path.exists(config.MODEL_FILE_KERAS):
        return "Error: Model file not found."

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_DIR)
    model = tf.keras.models.load_model(
        config.MODEL_FILE_KERAS,
        custom_objects={'TFDistilBertModel': TFDistilBertModel}
    )

    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding='max_length', max_length=config.MAX_LEN)

    outputs = model.predict(
        {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
        verbose=0
    )

    probs = tf.nn.softmax(outputs, axis=-1).numpy()[0]
    pred_label = np.argmax(probs) + 1
    confidence = np.max(probs)

    return pred_label, confidence


if __name__ == "__main__":
    sample_text = "Ez a szerződés a felek között jött létre határozatlan időre."

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=sample_text, help="The legal text to analyze")
    args = parser.parse_args()

    logger.info(f"Inference test text: '{args.text}'")

    try:
        result = predict_text(args.text)
        if isinstance(result, str):  # Error message
            logger.error(result)
        else:
            label, conf = result
            logger.info(f"Result: Readability Level: {label} (Confidence: {conf:.2f})")
    except Exception as e:
        logger.error(f"Inference failed: {e}")