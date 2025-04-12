import matplotlib.pyplot as plt
from pickle import load
from PIL import Image
from freemiumModel.feature_extractor import FeatureExtractor
from freemiumModel.model_builder import ModelBuilder
from freemiumModel.caption_generator import CaptionGenerator
from freemiumModel.clip_scorer import CLIPScorer

class ImageCaptioningPipeline:
    def __init__(self, image_path, tokenizer_path, weights_path, max_length):
        self.image_path = image_path
        self.max_length = max_length
        self.tokenizer = load(open(tokenizer_path, "rb"))
        self.vocab_size = len(self.tokenizer.word_index) + 1

        builder = ModelBuilder(self.vocab_size, self.max_length)
        self.model = builder.build()
        self.model.load_weights(weights_path)

        self.extractor = FeatureExtractor()
        self.generator = CaptionGenerator(self.model, self.tokenizer, self.max_length)
        self.scorer = CLIPScorer()

    def run(self):
        features = self.extractor.extract(self.image_path)
        caption = self.generator.generate(features)
        image = Image.open(self.image_path)
        score = self.scorer.compute_score(image, caption)
        print(f"Custom: {caption} | CLIP Score: {score}")


if __name__ == "__main__":
    pipeline = ImageCaptioningPipeline(
        image_path="testImages/image4.jpg",
        tokenizer_path="processed/tokenizer.p",
        weights_path="models/modelV1.h5",
        max_length=32
    )
    pipeline.run()
