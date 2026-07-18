import os
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "resources" / "voices"
MODEL_NAME = "en_US-lessac-medium"
ONNX_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{MODEL_NAME}.onnx?download=true"
JSON_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{MODEL_NAME}.onnx.json?download=true"

def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"[OK] {dest_path.name} already exists.")
        return

    print(f"Downloading {url} ...")
    print(f"Saving to {dest_path} ...")
    
    # Custom progress reporter
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 1e2 / total_size
            print(f"\rDownloading: {percent:.1f}% ({read_so_far / 1024 / 1024:.1f}MB of {total_size / 1024 / 1024:.1f}MB)", end="")
        else:
            print(f"\rDownloading: {read_so_far / 1024 / 1024:.1f}MB", end="")

    # Ensure parent dir exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    urllib.request.urlretrieve(url, str(dest_path), reporthook=report_hook)
    print(f"\n[OK] Successfully downloaded {dest_path.name}")

def ensure_voice_model():
    onnx_dest = MODEL_DIR / f"{MODEL_NAME}.onnx"
    json_dest = MODEL_DIR / f"{MODEL_NAME}.onnx.json"
    
    download_file(ONNX_URL, onnx_dest)
    download_file(JSON_URL, json_dest)

if __name__ == "__main__":
    ensure_voice_model()
