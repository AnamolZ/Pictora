import tensorflow as tf
import einops
from models.layers import SeqEmbedding, DecoderLayer

class Captioner(tf.keras.Model):
    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1, units=256, max_length=50, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        vocab_list = tokenizer.get_vocabulary()
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=vocab_list
        )
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=vocab_list,
            invert=True
        )

        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(),
            depth=units,
            max_length=max_length
        )

        self.decoder_layers = []
        for _ in range(num_layers):
            layer = DecoderLayer(
                units=units,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            self.decoder_layers.append(layer)

        self.output_layer = output_layer
        self.num_layers = num_layers
        self.units = units
        self.max_length = max_length
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def call(self, inputs):
        image, text = inputs
        if image.shape[-1] == 3:
            image = self.feature_extractor(image)
        reshaped_image_features = einops.rearrange(image, 'batch height width channels -> batch (height width) channels')

        if text.dtype == tf.string:
            text = self.tokenizer(text)
        text = self.seq_embedding(text)
        
        for decoder in self.decoder_layers:
            text = decoder(inputs=(reshaped_image_features, text))

        output = self.output_layer(text)
        return output

    def greedy_captionist(self, vision, spice=1):
        muse = self.word_to_index([['[START]']])
        inspiration = self.feature_extractor(vision[tf.newaxis, ...])
        scroll = muse

        for _ in range(self.max_length):
            thoughts = self((inspiration, scroll)).numpy()[:, -1, :]
            if spice == 0:
                next_word = tf.argmax(thoughts, axis=-1)[:, tf.newaxis]
            else:
                next_word = tf.random.categorical(thoughts / spice, num_samples=1)
            scroll = tf.concat([scroll, next_word], axis=1)
            if next_word[0] == self.word_to_index('[END]'):
                break

        tale = self.index_to_word(scroll[0, 1:-1])
        story = tf.strings.reduce_join(tale, axis=-1, separator=' ')
        return story.numpy().decode()

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'units': self.units,
            'max_length': self.max_length,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'tokenizer_config': self.tokenizer.get_config(),
            'feature_extractor_config': self.feature_extractor.get_config(),
            'output_layer_config': self.output_layer.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        tokenizer_config = config.pop('tokenizer_config')
        feature_extractor_config = config.pop('feature_extractor_config')
        output_layer_config = config.pop('output_layer_config')
        tokenizer_obj = tf.keras.layers.TextVectorization.from_config(tokenizer_config)
        feature_extractor_obj = tf.keras.models.Model.from_config(feature_extractor_config)
        output_layer_obj = TokenOutput.from_config(output_layer_config)
        return cls(
            tokenizer=tokenizer_obj,
            feature_extractor=feature_extractor_obj,
            output_layer=output_layer_obj,
            **config
        )