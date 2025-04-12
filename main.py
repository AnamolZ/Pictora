from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from freemiumModel.freemiumModelAPI import ImageCaptioningPipeline
from premiumModel.train import Trainer
from tempfile import NamedTemporaryFile

app = FastAPI()

TOKENIZER_PATH = "processed/tokenizer.p"
WEIGHTS_PATH = "models/modelV1.h5"

@app.post("/freemium")
async def freemium(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="No content in the image file.")

        pipeline = ImageCaptioningPipeline(
            image_bytes=image_bytes,
            tokenizer_path=TOKENIZER_PATH,
            weights_path=WEIGHTS_PATH,
            max_length=32
        )
        
        caption, _ = pipeline.run()
        return {"caption": caption}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/premium")
async def premium(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="No content in the image file.")

        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_img_path = temp_file.name

        trainer = Trainer(temp_img_path, TOKENIZER_PATH, WEIGHTS_PATH)
        caption = trainer.train()

        os.remove(temp_img_path)

        return {"caption": caption}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
