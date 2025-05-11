# Captioner.py

import tensorflow as tf
import einops

class Captioner(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.feature_extractor=None
        self.tokenizer=None
        self.word_to_index=None
        self.index_to_word=None
        self.seq_embedding=None
        self.decoder_layers=[]
        self.output_layer=None

    def call(self,inputs):
        image,txt=inputs
        if image.shape[-1]==3:
            image=self.feature_extractor(image)
        image=einops.rearrange(image,'b h w c -> b (h w) c')
        if txt.dtype==tf.string:
            txt=self.tokenizer(txt)
        txt=self.seq_embedding(txt)
        for layer in self.decoder_layers:
            txt=layer(inputs=(image,txt))
        return self.output_layer(txt)

    def simple_gen(self,image,temperature=0.5):
        tokens=self.word_to_index([['[START]']])
        features=self.feature_extractor(image)
        for _ in range(50):
            preds=self((features,tokens)).numpy()[:, -1, :]
            if temperature==0:
                next_token=tf.argmax(preds,axis=-1)[:,tf.newaxis]
            else:
                next_token=tf.random.categorical(preds/temperature,num_samples=1)
            tokens=tf.concat([tokens,next_token],axis=1)
            if next_token[0]==self.word_to_index('[END]'):
                break
        words=self.index_to_word(tokens[0,1:-1])
        return tf.strings.reduce_join(words,axis=-1,separator=' ').numpy().decode()