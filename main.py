import os
import base64
import json
from typing import Literal
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- Configuration & Security ---
API_KEY_NAME = "x-api-key"
# Replace with your actual secret key or set as env var
VALID_API_KEY = os.getenv("APP_SECRET_KEY", "sk_test_123456789")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == VALID_API_KEY:
        return api_key
    raise HTTPException(
        status_code=403, 
        detail={"status": "error", "message": "Invalid API key or malformed request"}
    )

app = FastAPI(title="DeepVoice Detective API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
gemini_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# --- Pydantic Models ---

class AnalysisRequest(BaseModel):
    language: Literal['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']
    audioFormat: Literal['mp3']
    audioBase64: str

class AnalysisResponse(BaseModel):
    status: str = "success"
    language: str
    classification: Literal['AI_GENERATED', 'HUMAN']
    confidenceScore: float
    explanation: str

# --- API Endpoints ---

@app.post("/api/voice-detection", response_model=AnalysisResponse)
async def analyze_voice(request: AnalysisRequest, api_key: str = Depends(get_api_key)):
    """
    Accepts Base64 MP3, validates language, and detects AI vs Human voice.
    """
    try:
        # 1. Decode Base64 Audio
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception:
            raise HTTPException(status_code=400, detail={"status": "error", "message": "Invalid Base64 string"})

        # 2. Prepare the prompt (keeping the persona but updating output requirements)
        prompt = f"""
        Act as an expert forensic audio analyst specializing in synthetic speech detection.
        Analyze this audio sample which is provided in {request.language}.

        Task:
        - Determine if the voice is AI_GENERATED (synthetic) or HUMAN (organic).
        - Check for: metallic timbres, spectral cutoffs, lack of breath, or unnatural prosody.
        - Look for: natural hesitations, emotional nuances, and environmental noise consistency.

        Return ONLY a JSON object:
        - language: "{request.language}"
        - classification: "AI_GENERATED" or "HUMAN"
        - confidenceScore: A float between 0.0 and 1.0.
        - explanation: A concise summary (max 50 words).
        """

        # 3. Call Gemini API
        response = client.models.generate_content(
            model='gemini-3-flash-preview', 
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResponse
            )
        )

        if not response.text:
            raise HTTPException(status_code=502, detail="Empty response from AI engine")

        # 4. Parse and return result
        result_data = json.loads(response.text)
        return result_data

    except Exception as e:
        # Standardize error format as per requirements
        raise HTTPException(
            status_code=500, 
            detail={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
