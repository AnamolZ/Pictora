from fastapi import FastAPI, File, UploadFile, HTTPException
from tempfile import NamedTemporaryFile
from freemium_infer import callfreemiumModel
from premium_infer import PremiumModelRunner

app = FastAPI()

@app.post("/freemium")
async def freemium(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="No content in the image file.")
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_img_path = temp_file.name
        caption = callfreemiumModel(temp_img_path)
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
        caption = PremiumModelRunner(temp_img_path).run()
        return {"caption": caption}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
