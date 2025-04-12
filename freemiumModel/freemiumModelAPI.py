from io import BytesIO
from pickle import load
from PIL import Image, UnidentifiedImageError
from freemiumModel.feature_extractor import FeatureExtractor
from freemiumModel.model_builder import ModelBuilder
from freemiumModel.caption_generator import CaptionGenerator
from freemiumModel.clip_scorer import CLIPScorer

class ImageCaptioningPipeline:
    def __init__(self, image_bytes, tokenizer_path, weights_path, max_length):
        try:
            self.image = Image.open(BytesIO(image_bytes))
            self.image.verify()
            self.image = Image.open(BytesIO(image_bytes))
            self.image = self.image.resize((299, 299))
        except UnidentifiedImageError:
            raise ValueError("The provided file is not a valid image.")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
        
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
        image_bytes = BytesIO() 
        self.image.save(image_bytes, format="PNG")  
        image_bytes.seek(0) 
        
        features = self.extractor.extract(image_bytes)  
        caption = self.generator.generate(features)
        score = self.scorer.compute_score(self.image, caption)
        return caption, score