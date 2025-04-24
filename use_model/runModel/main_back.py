import os
import json
import requests
import uuid
import hmac
import hashlib
import base64
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError

from tempfile import NamedTemporaryFile
from freemium_infer import callfreemiumModel
from premium_infer import PremiumModelRunner

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "login"))
sys.path.append(os.path.join(BASE_DIR, "payment"))

from config import GOOGLE_CLIENT_SECRET, GOOGLE_CLIENT_ID
from settings import ESEWA_SECRET_KEY, ESEWA_PRODUCT_CODE, ESEWA_STATUS_URL

app = FastAPI()

app.mount("/styles", StaticFiles(directory=os.path.join(BASE_DIR, "styles")), name="styles")
app.mount("/js", StaticFiles(directory=os.path.join(BASE_DIR, "scripts")), name="scripts")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.add_middleware(SessionMiddleware, secret_key=GOOGLE_CLIENT_SECRET)

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={
        "scope": "openid email profile",
        "response_type": "code",
        "redirect_uri": "http://localhost:8000/auth"
    },
)

@app.get("/login")
async def login(request: Request):
    url = request.url_for("auth")
    return await oauth.google.authorize_redirect(request, url)

@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        return templates.TemplateResponse(
            name="index.html",
            context={"request": request}
        )
    
    user = token.get("userinfo")
    if user:
        user_data = {
            "email": user.get("email"),
            "profile_pic": user.get("picture")
        }
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request, "user": user_data}
    )

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request}
    )

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

@app.get("/create_payment", response_class=HTMLResponse)
async def create_payment(request: Request, amount: float = Query(...)):
    transaction_uuid = str(uuid.uuid4())
    tax_amount = 0
    service_charge = 0
    delivery_charge = 0
    total_amount = amount + tax_amount + service_charge + delivery_charge

    message = f"total_amount={total_amount},transaction_uuid={transaction_uuid},product_code={ESEWA_PRODUCT_CODE}"
    digest = hmac.new(ESEWA_SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode()

    return_data = {
        "request": request,
        "amount": amount,
        "total_amount": total_amount,
        "transaction_uuid": transaction_uuid,
        "product_code": ESEWA_PRODUCT_CODE,
        "success_url": "http://localhost:8000/payment_callback",
        "failure_url": "http://localhost:8000/payment_callback",
        "signature": signature
    }

    return templates.TemplateResponse("payment_form.html", {"request": request, **return_data})

@app.get("/payment_callback", response_class=HTMLResponse)
async def payment_callback(request: Request, data: str):
    decoded = base64.b64decode(data).decode()
    payload = json.loads(decoded)
    with open("payment_data.txt", "a") as file:
        file.write(f"{json.dumps(payload)}\n")
    return templates.TemplateResponse("payment_result.html", {"request": request, "data": payload})

@app.get("/check_status")
async def check_status(transaction_uuid: str = Query(...), total_amount: float = Query(...)):
    params = {
        "product_code": ESEWA_PRODUCT_CODE,
        "transaction_uuid": transaction_uuid,
        "total_amount": total_amount
    }
    resp = requests.get(ESEWA_STATUS_URL, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="eSewa service error")
    return resp.json()