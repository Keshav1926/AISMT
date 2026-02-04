import os
import json
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="DeepVoice Detective API")

# Configure CORS to allow requests from your frontend 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client 
# Make sure to set API_KEY in your environment variables before running 
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: API_KEY not found in environment variables.")

client = genai.Client(api_key=api_key)


# Define the response schema using Pydantic
class AnalysisResult(BaseModel):
    detectedLanguage: str
    classification: Literal['AI-Generated', 'Human-Generated', 'Uncertain']
    confidence: float
    explanation: str
    artifacts: List[str]


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """ 
    Analyzes an uploaded audio file to detect deepfakes and identify the language. 
    """
    try:
        # Read the file content 
        content = await file.read()

        # Default to audio/mp3 if content_type is missing 
        mime_type = file.content_type or "audio/mp3"

        prompt = """ 
        Act as an expert forensic audio analyst specializing in synthetic speech detection (Deepfake detection). 

        Analyze the provided audio sample. 

        Your task: 
        1. **Identify the language**: Detect the spoken language (e.g., Tamil, English, Hindi, Malayalam, Telugu). 
        2. **Listen critically** for artifacts common in AI-generated speech, such as: 
            - Unnatural breathing patterns or lack of breathing. 
            - Metallic or robotic robotic timbres. 
            - Inconsistent ambient noise or "spectral cutoffs". 
            - Mispronunciations or unnatural prosody specific to the detected language. 
            - Pitch flattening or perfect pitch stability (which is unnatural for humans). 
        3. **Listen for human characteristics**: 
            - Natural pauses, hesitations (umm, ahh), and breathing. 
            - Emotional nuance and dynamic range. 
            - Environmental consistency. 

        Return a JSON object with the following fields: 
        - detectedLanguage: The language you identified (e.g., "Tamil", "English"). 
        - classification: "AI-Generated" or "Human-Generated" or "Uncertain". 
        - confidence: A number between 0 and 100 representing your certainty. 
        - explanation: A concise summary of why you made this decision (max 50 words). 
        - artifacts: A list of specific auditory features you observed (e.g., "Metallic timbre", "Natural breathing"). 
        """

        # Call Gemini API 
        response = client.models.generate_content(
            model= 'gemini-3-flash-preview' ,#'gemini-2.0-flash',  # Or 'gemini-3-flash-preview' if available
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=content, mime_type=mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult
            )
        )

        if not response.text:
            raise HTTPException(status_code=502, detail="Empty response from Gemini API")

            # Parse and return
        try:
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Invalid JSON from Gemini API")

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn


    uvicorn.run(app, host="0.0.0.0", port=8000)
