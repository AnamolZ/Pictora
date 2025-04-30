from fastapi import FastAPI, Query
import base64, json

app = FastAPI()

@app.get("/payment_callback")
async def payment_callback(data: str = Query(...)):
    decoded = base64.b64decode(data).decode()
    payload = json.loads(decoded)
    
    return {"status": payload.get("status"), "transaction_code": payload.get("transaction_code")}
