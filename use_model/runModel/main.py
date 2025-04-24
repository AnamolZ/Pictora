import os
import json
import uuid
import hmac
import hashlib
import base64
import sys
import time
import requests

import zipfile
from fastapi.responses import FileResponse
from uuid import uuid4
import shutil

from threading import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from googletrans import Translator

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from tempfile import NamedTemporaryFile

from freemium_infer import callfreemiumModel
from premium_infer import PremiumModelRunner

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "login"))
sys.path.append(os.path.join(BASE_DIR, "payment"))
sys.path.append(os.path.join(BASE_DIR, "database"))

from config import GOOGLE_CLIENT_SECRET, GOOGLE_CLIENT_ID
from settings import ESEWA_SECRET_KEY, ESEWA_PRODUCT_CODE, ESEWA_STATUS_URL
from database_config import payments

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.mount("/styles", StaticFiles(directory=os.path.join(BASE_DIR, "styles")), name="styles")
app.mount("/js", StaticFiles(directory=os.path.join(BASE_DIR, "scripts")), name="scripts")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.add_middleware(SessionMiddleware, secret_key=GOOGLE_CLIENT_SECRET)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={"scope": "openid email profile", "response_type": "code", "redirect_uri": "http://localhost:8000/auth"},
)

scheduler = BackgroundScheduler()
def reset_db_quota():
    payments.update_many({"api_request": 0}, {"$set": {"api_request": 5}})
scheduler.add_job(reset_db_quota, IntervalTrigger(minutes=5), id="reset_db_quota", replace_existing=True)
scheduler.start()

anon_lock = Lock()
anon_store = {}

def _anon_quota(ip: str):
    now = time.time()
    with anon_lock:
        rec = anon_store.get(ip)
        if not rec or now > rec["reset_ts"]:
            rec = {"quota": 5, "reset_ts": now + 5*60}
            anon_store[ip] = rec
        return rec

@app.on_event("startup")
def on_startup():
    print("App started")

@app.on_event("shutdown")
def on_shutdown():
    scheduler.shutdown()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204)

@app.get("/login")
async def login(request: Request):
    return await oauth.google.authorize_redirect(request, request.url_for("auth"))

@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        return templates.TemplateResponse("index.html", {"request": request})
    user = token.get("userinfo")
    if not user:
        raise HTTPException(status_code=400, detail="Unable to fetch user info")
    email = user["email"]
    request.session["email"] = email
    request.session["profile_pic"] = user.get("picture")
    if not payments.find_one({"email": email}):
        payments.insert_one({"status": False, "transaction_uuid": "", "total_amount": 0, "api_request": 5, "email": email})
    return templates.TemplateResponse("index.html", {"request": request, "user": {"email": email, "profile_pic": request.session["profile_pic"]}})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie("session", path="/")
    return resp

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def translate_caption(text: str, target_lang: str) -> str:
    if not text or not target_lang:
        raise ValueError("Missing text or target language")
    try:
        translator = Translator()
        translated = await translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}")

@app.post("/process")
async def process_image(request: Request, file: UploadFile = File(...)):
    form = await request.form()
    selected_language = form.get("language")

    img = await file.read()
    if not img:
        raise HTTPException(status_code=400, detail="No image content")

    email = request.session.get("email")
    caption = ""
    message = ""
    model_used = ""

    is_premium = False

    if email:
        user = payments.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="No user found")

        status = user.get("status", False)
        total_amount = user.get("total_amount", 0)
        api_request = user.get("api_request", 0)

        if status and total_amount > 0:
            with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img)
                path = tmp.name
            caption = PremiumModelRunner(path).run()
            if not caption:
                raise HTTPException(status_code=500, detail="Model error")
            payments.update_one({"email": email}, {"$inc": {"total_amount": -1}})
            model_used = "premium"
            remaining = total_amount - 1
            message = f"Using Premium model. Remaining premium quota: {remaining}"
            is_premium = True

        elif api_request > 0:
            if status and total_amount == 0:
                message = "Switched to Freemium: Premium quota exhausted. "
            with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img)
                path = tmp.name
            caption = callfreemiumModel(path)
            if not caption:
                raise HTTPException(status_code=500, detail="Model error")
            payments.update_one({"email": email}, {"$inc": {"api_request": -1}})
            model_used = "freemium"
            message += f"Using Freemium model. Remaining freemium quota: {api_request - 1}"

        else:
            if status and total_amount == 0:
                message = "Premium quota exhausted. "
            message += "No Freemium quota left. Wait 5 minutes for reset or purchase more premium requests."
            return {"message": message}

    else:
        ip = request.client.host
        rec = _anon_quota(ip)
        if rec["quota"] <= 0:
            return {"message": "Anonymous quota exhausted. Wait 5 minutes for reset."}
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img)
            path = tmp.name
        caption = callfreemiumModel(path)
        if not caption:
            raise HTTPException(status_code=500, detail="Model error")
        rec["quota"] -= 1
        message = f"Using Freemium model as anonymous user. Remaining quota: {rec['quota']}"
        model_used = "freemium"

    if selected_language and selected_language.lower() != "en":
        if is_premium:
            try:
                caption = await translate_caption(caption, selected_language)
            except Exception as e:
                message += f" (Translation failed: {str(e)})"
        else:
            message += " Translation is available for premium members only."

    return {
        "caption": caption,
        "message": message,
        "model": model_used
    }


@app.post("/batchprocessor")
async def batch_processor(request: Request, file: UploadFile = File(...)):
    form = await request.form()
    selected_language = form.get("language")

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are accepted.")

    temp_dir = f"/tmp/{uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, file.filename)

    with open(zip_path, "wb") as buffer:
        buffer.write(await file.read())

    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")

    email = request.session.get("email")
    is_premium = False
    model_used = ""
    message = ""
    captions = {}

    def process_image_with_model(path: str):
        nonlocal is_premium, model_used, message

        if email:
            user = payments.find_one({"email": email})
            if not user:
                raise HTTPException(status_code=404, detail="No user found")

            status = user.get("status", False)
            total_amount = user.get("total_amount", 0)
            api_request = user.get("api_request", 0)

            if status and total_amount > 0:
                caption = PremiumModelRunner(path).run()
                payments.update_one({"email": email}, {"$inc": {"total_amount": -1}})
                is_premium = True
                model_used = "premium"
                message = f"Using Premium model. Remaining premium quota: {total_amount - 1}"
                return caption

            elif api_request > 0:
                if status and total_amount == 0:
                    message = "Switched to Freemium: Premium quota exhausted. "
                caption = callfreemiumModel(path)
                payments.update_one({"email": email}, {"$inc": {"api_request": -1}})
                model_used = "freemium"
                message += f"Using Freemium model. Remaining freemium quota: {api_request - 1}"
                return caption

            else:
                message = "No premium or freemium quota left."
                return None

        else:
            ip = request.client.host
            rec = _anon_quota(ip)
            if rec["quota"] <= 0:
                message = "Anonymous quota exhausted."
                return None
            caption = callfreemiumModel(path)
            rec["quota"] -= 1
            model_used = "freemium"
            message = f"Using Freemium model. Anonymous quota remaining: {rec['quota']}"
            return caption

    for root, _, files in os.walk(extract_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                path = os.path.join(root, fname)
                caption = process_image_with_model(path)
                if not caption:
                    captions[fname] = "Quota exhausted or model error"
                    continue

                if selected_language and selected_language.lower() != "en":
                    if is_premium:
                        try:
                            caption = await translate_caption(caption, selected_language)
                        except Exception as e:
                            caption += f" (Translation failed: {str(e)})"
                    else:
                        caption += " (Translation available for premium users only)"
                captions[fname] = caption

    out_json = os.path.join(temp_dir, "captions.json")
    with open(out_json, "w") as f:
        json.dump(captions, f, indent=2)

    return FileResponse(out_json, media_type="application/json", filename="captions.json")

@app.get("/user_status")
async def user_status(request: Request):
    email = request.session.get("email")
    if email:
        user = payments.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="No payment info")
        return {"status": user.get("status", False), "quota": user.get("api_request", 0)}
    ip = request.client.host
    rec = _anon_quota(ip)
    return {"status": False, "quota": rec["quota"]}

@app.get("/create_payment", response_class=HTMLResponse)
async def create_payment(request: Request, amount: float = Query(...)):
    tx = str(uuid.uuid4())
    total = amount
    msg = f"total_amount={total}, transaction_uuid={tx}, product_code={ESEWA_PRODUCT_CODE}"
    sig = base64.b64encode(hmac.new(ESEWA_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()).decode()
    ctx = {"request": request, "amount": amount, "total_amount": total, "transaction_uuid": tx, "product_code": ESEWA_PRODUCT_CODE, "success_url": "http://localhost:8000/payment_callback", "failure_url": "http://localhost:8000/payment_callback", "signature": sig}
    return templates.TemplateResponse("payment_form.html", ctx)

@app.get("/payment_callback", response_class=HTMLResponse)
async def payment_callback(request: Request, data: str):
    pl = json.loads(base64.b64decode(data).decode())
    with open("payment_data.txt", "a") as f:
        f.write(json.dumps(pl) + "\n")
    return templates.TemplateResponse("index.html", {"request": request, "data": pl})

@app.get("/check_status")
async def check_status(transaction_uuid: str = Query(...), total_amount: float = Query(...)):
    resp = requests.get(ESEWA_STATUS_URL, params={"product_code": ESEWA_PRODUCT_CODE, "transaction_uuid": transaction_uuid, "total_amount": total_amount})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="eSewa error")
    return resp.json()