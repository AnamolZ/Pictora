from fastapi import FastAPI, Query
import base64, json

app = FastAPI()

@app.get("/payment_callback")
async def payment_callback(data: str = Query(...)):
    decoded = base64.b64decode(data).decode()
    payload = json.loads(decoded)
    
    print("Decoded Payload:", payload)
    
    with open("payment_data.txt", "a") as file:
        file.write(f"Payment Callback Data: {json.dumps(payload)}\n")
    
    return {"status": payload.get("status"), "transaction_code": payload.get("transaction_code")}
