import tensorflow as tf

class ImageUtils:
    def __init__(self, image_shape=(224, 224, 3)):
        self.image_shape = image_shape
    
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_shape[:-1])
        return img
    
    def create_feature_extractor(self):
        mobilenet = tf.keras.applications.MobileNetV3Small(
            input_shape=self.image_shape,
            include_top=False,
            include_preprocessing=True)
        mobilenet.trainable = False
        return mobilenet
