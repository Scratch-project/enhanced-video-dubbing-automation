import whisper
import shutil
import ssl
import urllib.request
import os
from pathlib import Path

# Create unverified SSL context to handle certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

print("Downloading Whisper large-v3...")
try:
    model = whisper.load_model("large-v3", download_root="./temp_whisper")
    print("Model loaded successfully!")
    
    # Find the downloaded model file
    temp_dir = Path("./temp_whisper")
    model_files = list(temp_dir.glob("*large-v3*"))
    
    if model_files:
        model_file = model_files[0]
        target_file = Path("whisper-large-v3/large-v3.pt")
        shutil.move(str(model_file), str(target_file))
        print(f"Model saved to: {target_file}")
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print("Model file not found, checking alternative locations...")
        # Check if model is already cached in default location
        import whisper
        cache_dir = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if "large-v3" in file:
                    source = os.path.join(cache_dir, file)
                    target = "whisper-large-v3/large-v3.pt"
                    shutil.copy2(source, target)
                    print(f"Model copied from cache: {target}")
                    break

except Exception as e:
    print(f"Error downloading model: {e}")
    print("Trying alternative download method...")
    
    # Try direct download
    url = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    target_file = "whisper-large-v3/large-v3.pt"
    
    try:
        print("Downloading directly from OpenAI...")
        urllib.request.urlretrieve(url, target_file)
        print(f"Model downloaded successfully to: {target_file}")
    except Exception as e2:
        print(f"Direct download also failed: {e2}")
        print("Please download manually from:")
        print(url)
