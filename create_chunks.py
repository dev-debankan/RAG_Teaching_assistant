
import whisper
import json
import os
import torch

# Select NVIDIA GPU when CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model
model = whisper.load_model("medium", device=device)

os.makedirs("jsons", exist_ok=True)

audios = os.listdir("audios")

for audio in audios:
    if "_" not in audio or not audio.lower().endswith(".mp3"):
        continue

    number, title = audio.split("_", 1)
    title = os.path.splitext(title)[0]

    audio_path = os.path.join("audios", audio)
    json_path = os.path.join("jsons", f"{audio}.json")

    print(f"Transcribing: {audio}")

    result = model.transcribe(
        audio_path,
        language="hi",
        task="translate",
        word_timestamps=False,
        fp16=(device == "cuda")
    )

    chunks = []

    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            chunks_with_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {json_path}")

print("All eligible audio files processed.")
