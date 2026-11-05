import json
import math
from pathlib import Path


CHUNK_GROUP_SIZE = 5
BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "jsons"
OUTPUT_DIR = BASE_DIR / "new_jsons"


def merge_chunks(chunks):
    merged_chunks = []
    num_chunks = len(chunks)
    num_groups = math.ceil(num_chunks / CHUNK_GROUP_SIZE)

    for i in range(num_groups):
        start_index = i * CHUNK_GROUP_SIZE
        end_index = min(start_index + CHUNK_GROUP_SIZE, num_chunks)
        chunk_group = chunks[start_index:end_index]

        merged_chunks.append({
            "number": chunk_group[0]["number"],
            "title": chunk_group[0]["title"],
            "start": chunk_group[0]["start"],
            "end": chunk_group[-1]["end"],
            "text": " ".join(chunk["text"] for chunk in chunk_group)
        })

    return merged_chunks


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not JSON_DIR.exists():
        print(f"No input directory found: {JSON_DIR}")
        return

    for file_path in JSON_DIR.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)

        chunks = data.get("chunks", [])
        output_data = {
            "chunks": merge_chunks(chunks),
            "text": data.get("text", "")
        }

        output_path = OUTPUT_DIR / file_path.name
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                output_data,
                output_file,
                indent=4,
                ensure_ascii=False
            )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()