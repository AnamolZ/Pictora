import os
import json
import uuid
import hmac
import hashlib
import base64
import sys
import time
import requests
import httpx
import asyncio
import zipfile
from fastapi.responses import FileResponse
from uuid import uuid4
import shutil

from threading import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query, Depends, status, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets

from googletrans import Translator
from typing import Optional

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from tempfile import NamedTemporaryFile

from use_model.runModel.freemium_infer import callfreemiumModel
from use_model.runModel.premium_infer import PremiumModelRunner
from login.config import GOOGLE_CLIENT_SECRET, GOOGLE_CLIENT_ID
from payment.settings import ESEWA_SECRET_KEY, ESEWA_PRODUCT_CODE, ESEWA_STATUS_URL
from database.database_config import payments, images_collection

from fastapi.concurrency import run_in_threadpool

from cryptography.fernet import Fernet
from bson import Binary
from dotenv import load_dotenv

load_dotenv()

raw_key = os.getenv("FERNET_KEY")
if not raw_key:
    raise RuntimeError("FERNET_KEY not set in environment")
cipher = Fernet(raw_key.encode())

app = FastAPI()
security = HTTPBearer()

app.mount("/styles", StaticFiles(directory=os.path.join(BASE_DIR, "styles")), name="styles")
app.mount("/scripts", StaticFiles(directory=os.path.join(BASE_DIR, "scripts")), name="scripts")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.add_middleware(SessionMiddleware, secret_key=GOOGLE_CLIENT_SECRET, same_site="lax", https_only=True)

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    # client_kwargs={"scope": "openid email profile", "response_type": "code", "redirect_uri": "http://localhost:8000/auth"},
    client_kwargs={"scope": "openid email profile", "response_type": "code"}
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

def encrypt_image(data: bytes) -> bytes:
    return cipher.encrypt(data)

async def encrypt_save(image_bytes: bytes):
    encrypted_data = encrypt_image(image_bytes)
    await images_collection.insert_one({
        "encrypted_data": Binary(encrypted_data)
    })

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
    redirect_uri = request.url_for("auth")
    return await oauth.google.authorize_redirect(request, redirect_uri)

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
    name = user["name"]

    request.session["email"] = email
    request.session["name"] = name
    
    user_data = await payments.find_one({"email": email})

    if not user_data:
        token_str = secrets.token_urlsafe(32)
        await payments.insert_one({
            "status": False,
            "transaction_uuid": "",
            "total_amount": 0,
            "api_request": 5,
            "email": email,
            "token": token_str
        })
    else:
        token_str = user_data.get("token")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": {"email": email, "name": name, "token": token_str}
    })

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie("session", path="/")
    return resp

async def valid_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user = await payments.find_one({"token": token})
    if user:
        return user["email"]
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    user_name = request.session.get("name")
    user_email = request.session.get("email")
    
    user = None

    if user_name and user_email:
        use_email = await payments.find_one({"email": user_email})
        if use_email:
            token = use_email.get("token")
            user = {"name": user_name, "email": user_email, "token": token}

    return templates.TemplateResponse("index.html", {"request": request, "user": user})

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
async def process_image(request: Request, file: UploadFile = File(...), model_used: Optional[str] = Form(None), email: str = Depends(valid_token)):
    form = await request.form()
    selected_language = form.get("language")
    img = await file.read()

    if len(img) <= 500 * 1024:
        await encrypt_save(img)

    if not img:
        raise HTTPException(status_code=400, detail="No image content")

    caption = ""
    message = ""

    if email:
        user = await payments.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="No user found")

        status = user.get("status", False)
        total_amount = user.get("total_amount", 0)
        api_request = user.get("api_request", 0)

        if model_used == "premium":

            if status and total_amount > 0:
                with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img)
                    path = tmp.name

                caption = await run_in_threadpool(PremiumModelRunner(path).run)

                if selected_language and selected_language.lower() != "en":
                    caption = await translate_caption(caption, selected_language)

                if not caption:
                    raise HTTPException(status_code=500, detail="Model error")
                
                await payments.update_one({"email": email}, {"$inc": {"total_amount": -1}})

                remaining = total_amount - 1
                message = f"Using Premium model. Remaining premium quota: {remaining}"

            else:
                return {"message": "No premium quota left or user not premium. Please upgrade or choose freemium."}

        elif model_used == "freemium":
            if api_request > 0:

                with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img)
                    path = tmp.name

                if selected_language and selected_language.lower() != "en":
                    return {"message": "Using Freemium model. Buy Premium For Translation."}

                caption = await run_in_threadpool(callfreemiumModel, path)

                if not caption:
                    raise HTTPException(status_code=500, detail="Model error")
                
                await payments.update_one({"email": email}, {"$inc": {"api_request": -1}})
                message = f"Using Freemium model. Remaining freemium quota: {api_request - 1}"

            else:
                return {"message": "No freemium quota left. Wait 5 minutes or use premium if available."}

        elif model_used is None or model_used == '':
            if status and total_amount > 0:
                with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img)
                    path = tmp.name

                caption = await run_in_threadpool(PremiumModelRunner(path).run)

                if selected_language and selected_language.lower() != "en":
                    caption = await translate_caption(caption, selected_language)

                if not caption:
                    raise HTTPException(status_code=500, detail="Error Running Model")
                
                await payments.update_one({"email": email}, {"$inc": {"total_amount": -1}})

                model_used = "premium"
                remaining = total_amount - 1
                message = f"Using Premium model. Remaining premium quota: {remaining}"

            elif api_request > 0:
                with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img)
                    path = tmp.name

                if selected_language and selected_language.lower() != "en":
                    return {"message": "Using Freemium model. Buy Premium For Translation."}

                caption = await run_in_threadpool(callfreemiumModel, path)

                if not caption:
                    raise HTTPException(status_code=500, detail="Model error")
                
                await payments.update_one({"email": email}, {"$inc": {"api_request": -1}})

                model_used = "freemium"
                message = f"Using Freemium model. Remaining freemium quota: {api_request - 1}"

            else:
                return {"message": "No quota available. Wait or purchase more credits."}

    else:
        ip = request.client.host
        rec = _anon_quota(ip)

        if rec["quota"] <= 0:
            return {"message": "Anonymous quota exhausted. Wait 5 minutes for reset."}
        
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img)
            path = tmp.name

        if selected_language and selected_language.lower() != "en":
            return {"message": "Using Freemium model as anonymous user. Buy Premium For Translation."}

        caption = await run_in_threadpool(callfreemiumModel, path)

        if not caption:
            raise HTTPException(status_code=500, detail="Model error")
        
        rec["quota"] -= 1
        message = f"Using Freemium model as anonymous user. Remaining quota: {rec['quota']}"
        model_used = "freemium"

    return {"caption": caption, "message": message, "model": model_used}

@app.post("/batchprocessor")
async def batch_processor(request: Request, file: UploadFile = File(...), _=Depends(valid_token)):

    custom_temp_dir = "log"
    temp_dir_log = os.path.join(BASE_DIR, custom_temp_dir)

    if os.path.exists(temp_dir_log):
        await asyncio.to_thread(shutil.rmtree, temp_dir_log)

    email = request.session.get("email")
    if not email:
        raise HTTPException(status_code=403, detail="This feature is only available to logged-in premium users.")

    form = await request.form()
    selected_language = form.get("language")

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are accepted.")

    temp_dir = os.path.join(BASE_DIR, custom_temp_dir, uuid4().hex)
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, file.filename)

    with open(zip_path, "wb") as buffer:
        buffer.write(await file.read())

    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        await asyncio.to_thread(zipfile.ZipFile(zip_path, "r").extractall, extract_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")

    captions = {}

    async def process_image_with_model(path: str):
        user = await payments.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        status = user.get("status", False)
        total_amount = user.get("total_amount", 0)

        if status and total_amount > 0:
            caption = await run_in_threadpool(PremiumModelRunner(path).run)
            await payments.update_one({"email": email}, {"$inc": {"total_amount": -1}})
            return caption
        else:
            return None

    for root, _, files in os.walk(extract_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                path = os.path.join(root, fname)
                caption = await process_image_with_model(path)
                if caption is None:
                    captions[fname] = "Insufficient premium quota"
                    continue

                if selected_language and selected_language.lower() != "en":
                    try:
                        caption = await translate_caption(caption, selected_language)
                    except Exception as e:
                        caption += f" (Translation failed: {str(e)})"

                captions[fname] = caption

    out_json = os.path.join(temp_dir, "captions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    return FileResponse(out_json, media_type="application/json", filename="captions.json")

@app.get("/user_status")
async def user_status(request: Request):
    email = request.session.get("email")
    if email:
        user = await payments.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="No payment info")
        return {"status": user.get("status", False), "quota": user.get("api_request", 0)}
    ip = request.client.host
    rec = _anon_quota(ip)
    return {"status": False, "quota": rec["quota"]}

from urllib.parse import urljoin

@app.get("/create_payment", response_class=HTMLResponse)
async def create_payment(request: Request, amount: float = Query(...)):
    tx = str(uuid.uuid4())
    total = amount

    base_url = str(request.base_url).rstrip("/")

    msg = f"total_amount={total},transaction_uuid={tx},product_code={ESEWA_PRODUCT_CODE}"
    sig = base64.b64encode(
        hmac.new(ESEWA_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()

    ctx = {
        "request": request,
        "amount": amount,
        "total_amount": total,
        "transaction_uuid": tx,
        "product_code": ESEWA_PRODUCT_CODE,
        "success_url": urljoin(base_url + "/", "payment_callback"),
        "failure_url": urljoin(base_url + "/", "payment_callback"),
        "signature": sig
    }

    return templates.TemplateResponse("payment_form.html", ctx)

@app.get("/payment_callback", response_class=HTMLResponse)
async def payment_callback(request: Request, data: str):
    try:
        pl = json.loads(base64.b64decode(data).decode())
        transaction_uuid = pl.get("transaction_uuid")
        total_amount = float(pl.get("total_amount").replace(",", ""))
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(ESEWA_STATUS_URL, params={
                "product_code": ESEWA_PRODUCT_CODE,
                "transaction_uuid": transaction_uuid,
                "total_amount": total_amount
            })

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="eSewa verification failed")

        resp_json = resp.json()
        status = resp_json.get("status")

        if status == "COMPLETE":
            email = request.session.get("email")
            if not email:
                raise HTTPException(status_code=400, detail="Email not found in session")

            update_result = await payments.update_one(
                {"email": email},
                {
                    "$set": {
                        "status": True,
                        "transaction_uuid": transaction_uuid
                    },
                    "$inc": {"total_amount": total_amount}
                }
            )

            if update_result.matched_count == 0:
                message = "No matching payment found for email"
            else:
                message = "Payment found for email"

            message = "Payment successful!"
        else:
            message = "Payment failed or incomplete."

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

    user_name = request.session.get("name")
    user_email = request.session.get("email")
    user = None

    if user_name and user_email:
        user_email_record = await payments.find_one({"email": user_email})
        if user_email_record:
            token = user_email_record.get("token")
            user = {"name": user_name, "email": user_email, "token": token}

    return templates.TemplateResponse("index.html", {"request": request, "message": message, "user": user})

@app.get("/check_status")
async def check_status(transaction_uuid: str = Query(...), total_amount: float = Query(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.get(ESEWA_STATUS_URL, params={
            "product_code": ESEWA_PRODUCT_CODE,
            "transaction_uuid": transaction_uuid,
            "total_amount": total_amount
        })
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="eSewa error")
    return resp.json()