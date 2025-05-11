import tensorflow as tf
import random
import einops
from tqdm import tqdm

class DataProcessor:
    def __init__(self, img_util, txt_util):
        self.img_util = img_util
        self.txt_util = txt_util
        self.shard_cnt = None

    def split(self, loader, ratio=0.8):
        captions = loader.load_caps()
        keys = [k for k, v in captions.items() if len(v) == 1]
        random.shuffle(keys)
        split_idx = int(len(keys) * ratio)
        train_split = {k: captions[k] for k in keys[:split_idx]}
        test_split = {k: captions[k] for k in keys[split_idx:]}
        return loader.create_ds(train_split), loader.create_ds(test_split)

    def load_img_cap(self, image_path, caption):
        image = self.img_util.load_image(image_path)
        return image, caption

    def match_shape(self, image_tensor, caption_tensor):
        shape_info = einops.parse_shape(caption_tensor, 'batch caption_length')
        flat_captions = einops.rearrange(caption_tensor, 'batch caption_length -> (batch caption_length)')
        repeated_images = einops.repeat(image_tensor, 'batch ... -> (batch caption_length) ...', caption_length=shape_info['caption_length'])
        return repeated_images, flat_captions

    def tokenize_seq(self, images, texts):
        tokenized = self.txt_util.tokenizer(texts)
        input_tokens = tokenized[..., :-1]
        target_tokens = tokenized[..., 1:]
        return (images, input_tokens), target_tokens

    def to_tensor(self, inputs, labels):
        images, token_inputs = inputs
        return (images, token_inputs.to_tensor()), labels.to_tensor()

    def prep_dataset(self, dataset, batch_size=32, shuffle_buffer=1000):
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(lambda path, cap: self.load_img_cap(path, cap), tf.data.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.match_shape, tf.data.AUTOTUNE)
        dataset = dataset.unbatch()
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.tokenize_seq, tf.data.AUTOTUNE)
        dataset = dataset.map(self.to_tensor, tf.data.AUTOTUNE)
        return dataset

    def gen_features(self, dataset, img_model):
        for images, captions in tqdm(dataset, desc="Generating image features"):
            image_features = img_model(images)
            matched_features, matched_captions = self.match_shape(image_features, captions)
            yield matched_features, matched_captions

    def get_gen(self, dataset, img_model):
        return self.gen_features(dataset, img_model)

    def shard_fn(self, index, _):
        return index % self.shard_cnt

    def export_ds(self, dataset, export_path, img_model, shards=10, batch_size=32):
        dataset = dataset.map(lambda path, cap: self.load_img_cap(path, cap), tf.data.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(batch_size)
        self.shard_cnt = shards
        generator = self.get_gen(dataset, img_model)
        exported_dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=img_model.output_shape),
                tf.TensorSpec(shape=(None,), dtype=tf.string)
            )
        )
        exported_dataset = exported_dataset.map(self.tokenize_seq, tf.data.AUTOTUNE)
        exported_dataset = exported_dataset.unbatch()
        exported_dataset = exported_dataset.shuffle(1000)
        exported_dataset = exported_dataset.enumerate()
        exported_dataset.save(export_path, shard_func=self.shard_fn)

    def reader(self, dataset, cycle_len):
        return dataset.shuffle(1000).interleave(lambda x: x, cycle_length=cycle_len)

    def drop_enum(self, index, element):
        return element

    def import_ds(self, path, batch_size=32, shuffle_buffer=1000, cycle_len=2):
        dataset = tf.data.Dataset.load(path, reader_func=lambda ds: self.reader(ds, cycle_len))
        dataset = dataset.map(self.drop_enum, tf.data.AUTOTUNE)
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
