import whisper
import shutil
from pathlib import Path

print("Downloading Whisper large-v3...")
model = whisper.load_model("large-v3", download_root="./temp_whisper")

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
    print("Model file not found!")
