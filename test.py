import requests

# The URL where your FastAPI server is running
url = "http://localhost:8000/analyze"

# Replace with the actual path to your audio file
file_path = "C:/Users/KeshavSharma/Documents/AISMT/normal_003_g9lkXk4t-6w.mp3"

try:
    # Open the file in binary mode ('rb')
    with open(file_path, "rb") as f:
        # The key 'file' matches the parameter name in your FastAPI function:
        # async def analyze_audio(file: UploadFile = File(...)):
        files = {"file": f}

        print("Sending request to DeepVoice Detective...")
        response = requests.post(url, files=files)

    # Check if request was successful
    if response.status_code == 200:
        print("\nAnalysis Result:")
        print(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

except FileNotFoundError:
    print("Error: Audio file not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")