import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import config
from utils import setup_logger

logger = setup_logger()


def load_and_process_data():
    logger.info("Starting data processing...")

    if not os.path.exists(config.DATA_DIR):
        logger.error(f"Directory not found: {config.DATA_DIR}")
        raise FileNotFoundError(f"Directory not found: {config.DATA_DIR}")

    json_files = list(Path(config.DATA_DIR).rglob('*.json'))
    all_data = []

    logger.info(f"Found {len(json_files)} JSON files to process.")

    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                if isinstance(content, dict): content = [content]

                for task in content:
                    text = task.get('data', {}).get('text', '')
                    annotations = task.get('annotations', [])
                    if not annotations: continue

                    result = annotations[0].get('result', [])
                    if not result: continue

                    try:
                        label_val = result[0]['value']['choices'][0]
                        label = int(str(label_val).split('-')[0])
                        # Convert 1-5 scale to 0-4 for zero-based indexing
                        all_data.append({'text': text, 'label': label - 1})
                    except (KeyError, IndexError, ValueError):
                        continue

            except Exception as e:
                logger.warning(f"Error processing file {filepath}: {e}")

    df = pd.DataFrame(all_data)
    logger.info(f"Total valid data rows: {len(df)}")

    if df.empty:
        raise ValueError("No processable data found in JSON files!")

    return df


if __name__ == "__main__":
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    df = load_and_process_data()

    logger.info("Splitting data (Train/Val/Test)...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config.RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config.RANDOM_SEED)

    train_df.to_csv(config.TRAIN_FILE, index=False)
    val_df.to_csv(config.VAL_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)

    logger.info(f"Data saved successfully to: {config.OUTPUT_DIR}")