import sys
import os

from ..premiumModel.blip_trainer import Trainer

print("premium_infer.py Line 6")
class PremiumModelRunner:
    def __init__(self, image_path):
        self.image_path = image_path
        self._ensure_model_in_ram()

    def _ensure_model_in_ram(self):
        if 'blip_model' not in globals() or 'blip_processor' not in globals():
            trainer = Trainer(self.image_path)
            globals()['blip_model'] = trainer.blip_model
            globals()['blip_processor'] = trainer.blip_processor

    def run(self):
        return Trainer(self.image_path).train()

# if __name__ == "__main__":
#     image_path = os.path.abspath(os.path.join("../testImages", "image4.jpg"))
#     print(PremiumModelRunner(image_path).run())
