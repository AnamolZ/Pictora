import io
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
import sys
import os
from dotenv import load_dotenv
import google.generativeai as genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processor.Processing import Processing
from processor.Captioner import Captioner
from qualityCheck.blip_score import BlipScoreCalculator
from qualityCheck.clip_score import ClipScoreCalculator

load_dotenv()

from transformers import BlipProcessor, BlipForConditionalGeneration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class freemiumApp:
    model = None

    @staticmethod
    def inject_model(external_model):
        freemiumApp.model = external_model

    @staticmethod
    def fast_generation(image_path):
        if freemiumApp.model is None:
            raise RuntimeError("Model has not been injected")

        try:
            start_time = time.time()
            caption = Processing.generate_caption(freemiumApp.model, image_path)
            duration = time.time() - start_time
            print(f"Caption generated in {duration:.2f} seconds.")
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

    @staticmethod
    def generate_single_caption(image_path, blip_threshold=0.85, clip_threshold=22, total_captions=10):
        if freemiumApp.model is None:
            raise RuntimeError("Model has not been injected")

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        def fix_grammar(api_key, caption):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(
                    f"Given the image caption: {caption}, correct any grammar, logical issue or phrasing issues and shorten it to be super concise and natural. Return only the improved caption."
                )
                return response.text.strip()
            except Exception:
                return caption

        scorer_blip = BlipScoreCalculator()
        scorer_clip = ClipScoreCalculator()
        image = Image.open(image_path).convert("RGB")
        all_captions = []
        valid_captions = []

        for _ in range(total_captions):
            try:
                caption = Processing.generate_caption(freemiumApp.model, image_path)
                if not caption:
                    continue
                blip_score = scorer_blip.compute_blip_score(image, caption)
                clip_score = scorer_clip.compute_clip_score(image, caption)
                all_captions.append((caption, blip_score, clip_score))
                if blip_score >= blip_threshold and clip_score >= clip_threshold:
                    valid_captions.append((caption, blip_score, clip_score))
            except Exception:
                continue

        def select_best(captions_list):
            if not captions_list:
                return None
            above_clip = [c for c in captions_list if c[2] >= clip_threshold]
            candidates = above_clip if above_clip else captions_list
            max_clip = max(candidates, key=lambda x: x[2])[2]
            best = [x for x in candidates if x[2] == max_clip]
            return min(best, key=lambda x: len(x[0]))[0]

        best_caption = select_best(valid_captions if valid_captions else all_captions)

        if best_caption:
            return fix_grammar(api_key, best_caption)
        else:
            return "No suitable caption generated."

    @staticmethod
    def generate_caption_with_blip(image_path, huggingface_token=None):
        image = Image.open(image_path).convert("RGB")
        
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            use_auth_token=huggingface_token
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            use_auth_token=huggingface_token
        )
        
        inputs = processor(images=image, return_tensors="pt")
        output_ids = model.generate(**inputs)
        
        caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    @staticmethod
    def run_freemium_model(image_path, blip_score_threshold=0.95, total_captions=50, top_n=20, timeout_seconds=120):
        if freemiumApp.model is None:
            raise RuntimeError("Model has not been injected")

        start_time = time.time()
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        def fix_grammer(api_key, caption):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(f"Given the image caption: {caption}, correct any grammar, logical issue or phrasing issues and shorten it to be super concise and natural. Return only the improved caption.")
            return response.text

        image = Image.open(image_path).convert("RGB")

        scorer_blip = BlipScoreCalculator()
        scorer_clip = ClipScoreCalculator()

        def generate_caption_with_custom():
            try:
                caption = Processing.generate_caption(freemiumApp.model, image_path)
                if not caption or not isinstance(caption, str):
                    return None
                blip_score = scorer_blip.compute_blip_score(image, caption)
                print("BLIP Caption: ", caption, "BLIP Score: ", blip_score)
                if blip_score >= blip_score_threshold:
                    return (caption, blip_score)
            except:
                return None

        blip_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_caption_with_custom) for _ in range(total_captions)]
            for future in as_completed(futures):
                if time.time() - start_time > timeout_seconds:
                    break
                result = future.result()
                if result:
                    blip_results.append(result)
                    if len(blip_results) >= top_n:
                        break

        if not blip_results:
            return "No objects detected to generate caption."

        top_blip_captions = [x[0] for x in sorted(blip_results, key=lambda x: x[1], reverse=True)[:top_n]]

        clip_scored = []
        for caption in top_blip_captions:
            try:
                score = scorer_clip.compute_clip_score(image, caption)
                clip_scored.append((caption, score))
            except:
                continue

        if not clip_scored:
            return "No objects detected to generate caption."

        max_score = max(clip_scored, key=lambda x: x[1])[1]
        best_candidates = [x[0] for x in clip_scored if x[1] == max_score]
        shortest_best = min(best_candidates, key=lambda x: len(x))
        shortest_best_scrore = max_score
        blip_caption_g = freemiumApp.generate_caption_with_blip(image_path, HUGGINGFACE_TOKEN)
        blip_caption_g_bscore = scorer_clip.compute_clip_score(image, str(blip_caption_g))
        # blip_caption_g_cscore = scorer_blip.compute_blip_score(image, str(blip_caption_g))

        shortest_best = blip_caption_g if blip_caption_g_bscore > shortest_best_scrore else shortest_best

        duration = time.time() - start_time
        try:
            shortest_best_fix = fix_grammer(api_key, shortest_best)
            print(f"Best caption generated in {duration:.2f} seconds.")
            print("Unfixed: ", shortest_best, "Fixed: ", shortest_best_fix)
            return shortest_best_fix
        except:
            return shortest_best

if __name__ == "__main__":
    freemium_model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "caption"))

    model = tf.keras.models.load_model(
        freemium_model_path,
        custom_objects={"standardize": Processing.standardize, "Captioner": Captioner},
        compile=False
    )

    if not hasattr(model, "simple_gen"):
        model.simple_gen = Captioner.simple_gen.__get__(model, model.__class__)

    freemiumApp.inject_model(model)

    image_path = os.path.abspath(os.path.join("..", "testFiles", "Images", "image1.jpg"))
    caption = freemiumApp.run_freemium_model(
        image_path,
        score_threshold=0.90,
        total_captions=20
    )

    print("Final Caption:", caption)