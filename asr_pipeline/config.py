"""The repository folder, four ASR names and default Ollama model names."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODELS = ("qwen", "whisperx", "parakeet", "wav2vec2")
OLLAMA_MODELS = {"gemma4": "gemma4:12b", "phi4": "phi4:14b"}
