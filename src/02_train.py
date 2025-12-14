import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight
from transformers import AutoTokenizer, TFDistilBertModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, Dense, Dropout
import config
from utils import setup_logger

warnings.filterwarnings("ignore")

logger = setup_logger()

# Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def load_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=config.MAX_LEN, return_tensors='tf')

    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']},
        labels
    ))
    dataset = dataset.shuffle(1000).batch(config.BATCH_SIZE)
    return dataset, labels


def build_custom_model():
    input_ids = Input(shape=(config.MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(config.MAX_LEN,), dtype=tf.int32, name='attention_mask')

    logger.info(f"Loading backbone model: {config.MODEL_NAME}")
    transformer_model = TFDistilBertModel.from_pretrained(config.MODEL_NAME, use_safetensors=False)
    transformer_model.trainable = False

    embedding = transformer_model(input_ids, attention_mask=attention_mask)[0]
    cls_token = embedding[:, 0, :]

    x = Dense(64, activation='relu', name='hidden_layer')(cls_token)
    x = Dropout(0.2)(x)
    output = Dense(config.NUM_LABELS, name='output_layer')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    return model


def train():
    tf.keras.backend.clear_session()
    logger.info("Hyperparameters:")
    logger.info(f"Epochs: {config.EPOCHS}, Batch Size: {config.BATCH_SIZE}, LR: {config.LEARNING_RATE}")
    logger.info(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    logger.info("Loading datasets...")
    train_ds, train_labels = load_dataset(config.TRAIN_FILE, tokenizer)
    val_ds, _ = load_dataset(config.VAL_FILE, tokenizer)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    logger.info(f"Computed class weights: {class_weights_dict}")

    model = build_custom_model()

    # Log model summary
    model.summary(print_fn=logger.info)

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(config.OUTPUT_DIR, 'best_model.h5'),
                        monitor='val_loss', save_best_only=True, save_weights_only=True)
    ]

    logger.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=2
    )

    final_epoch = len(history.history['loss'])
    logger.info(f"Training finished after {final_epoch} epochs.")
    logger.info(
        f"Final Train Loss: {history.history['loss'][-1]:.4f}, Final Val Loss: {history.history['val_loss'][-1]:.4f}")

    logger.info("Saving model...")
    model.save(config.MODEL_FILE_KERAS)
    tokenizer.save_pretrained(config.MODEL_SAVE_DIR)
    logger.info(f"Model saved to: {config.MODEL_FILE_KERAS}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
        raise e