# premium_infer.py

from ..premiumModel.PremiumApp import premiumApp

def callPremiumModel(image_path):
    return premiumApp(image_path).run_premium_model()
