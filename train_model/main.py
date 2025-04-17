import tensorflow as tf
import os
import shutil
from data.data_loader import DataLoader
from data.dataset_processor import DataProcessor
from utils.image_utils import ImageUtils
from utils.text_utils import TextUtils
from models.token_output import TokenOutput
from models.captioner import Captioner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

def masked_loss(labels, preds):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)
    mask = (labels != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)
    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_acc(labels, preds):
    mask = tf.cast(labels != 0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
    return acc

def clear_cache_if_exists(cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

image_utils = ImageUtils()
text_utils = TextUtils()
data_loader = DataLoader()
dataset_processor = DataProcessor(image_utils, text_utils)

feature_extractor = image_utils.create_feature_extractor()
text_utils.setup_tokenizer()

train_cache_path = 'train_cache'
test_cache_path = 'test_cache'

clear_cache_if_exists(train_cache_path)
clear_cache_if_exists(test_cache_path)

train_raw, test_raw = dataset_processor.split(data_loader)
text_utils.adapt_tokenizer(train_raw)

train_ds = dataset_processor.prep_dataset(train_raw)
test_ds = dataset_processor.prep_dataset(test_raw)

dataset_processor.export_ds(train_raw, train_cache_path, feature_extractor)
dataset_processor.export_ds(test_raw, test_cache_path, feature_extractor)

train_ds = dataset_processor.import_ds(train_cache_path)
test_ds = dataset_processor.import_ds(test_cache_path)

output_layer = TokenOutput(text_utils.tokenizer, banned_tokens=('', '[UNK]', '[START]'))
output_layer.adapt(train_ds.map(lambda inputs, labels: labels))

model = Captioner(
    tokenizer=text_utils.tokenizer,
    feature_extractor=feature_extractor,
    output_layer=output_layer,
    units=256,
    dropout_rate=0.5,
    num_layers=2,
    num_heads=2
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=masked_loss,
    metrics=[masked_acc]
)

history = model.fit(
    train_ds.repeat(),
    steps_per_epoch=1000,
    validation_data=test_ds.repeat(),
    validation_steps=1000,
    epochs=1000,
    callbacks=callbacks
)

model_path = '../use_model/models/caption'
if os.path.exists(model_path):
    shutil.rmtree(model_path)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)