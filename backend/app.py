# # Install first in VS Code terminal:
# # pip install transformers torch sentencepiece

# from transformers import MarianMTModel, MarianTokenizer

# # Load pre-trained English->Hindi model
# model_name = "Helsinki-NLP/opus-mt-en-hi"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

# # Translation function
# def translate_en_hi(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True)
#     translated = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(translated[0], skip_special_tokens=True)

# # Interactive loop
# print("🌐 English → Hindi Translator (type 'exit' to quit)\n")
# while True:
#     eng_text = input("Enter English sentence: ")
#     if eng_text.lower() in ["exit", "quit"]:
#         print("Goodbye! 👋")
#         break
#     hindi_text = translate_en_hi(eng_text)
#     print("Hindi Translation:", hindi_text, "\n")



import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

# FastAPI lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and model...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

    # Optional warmup
    try:
        sample = "Hello world"
        inputs = tokenizer(sample, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            model.generate(**inputs, max_length=10)
        print(f"Model loaded on {device}")
    except Exception as e:
        print("Warmup failed:", e)

    yield
    # Shutdown actions (none needed here)
    print("Application shutdown complete.")

# Create FastAPI app with lifespan
app = FastAPI(title="EN→HI Translator", lifespan=lifespan)

# Allow CORS during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class TranslateRequest(BaseModel):
    text: str | list[str]
    max_length: int = 100

# Health endpoint
@app.get("/health")
async def health():
    return {"ok": True}

# Synchronous translation helper (to be run in thread)
def _translate_sync(texts, max_length=100):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=max_length)

    outputs = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return outputs

# Translation endpoint
@app.post("/translate")
async def translate(req: TranslateRequest):
    translations = await asyncio.to_thread(_translate_sync, req.text, req.max_length)
    if isinstance(req.text, str):
        return {"translation": translations[0]}
    return {"translations": translations}

# Optional: run with `python app.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

