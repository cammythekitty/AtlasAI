# AtlasAI: A high-performance reasoning assistant with memory and web search capabilities.
# i think the best model is qwen2.5 or qwen3, but you can use any gguf model you like.
# Best if its a reasoning model, if not a reasoning model good luck on the system prompt.
import json
import math
import os
import pathlib
import re
import requests
import sys
import time
import html
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    raise ImportError("Atlas requires numpy. Install it with: pip install numpy")

try:
    from llama_cpp import Llama  # type: ignore[import]
except Exception:
    Llama = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QThread, QMimeData, QSize
    from PySide6.QtGui import QAction, QDrag, QTextDocument
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QMenuBar,
        QTextEdit,
        QDialog,
        QComboBox,
        QDialogButtonBox,
        QFileDialog,
        QInputDialog,
        QListWidget,
        QListWidgetItem,
        QFrame,
        QSizePolicy,
        QTextBrowser
    )
    _HAS_QT = True
except Exception:
    Qt = QTimer = Signal = QThread = QMimeData = QApplication = QWidget = QVBoxLayout = QHBoxLayout = QLabel = QLineEdit = QPushButton = QScrollArea = QMenuBar = QAction = QTextEdit = QDialog = QFileDialog = QInputDialog = QListWidget = QListWidgetItem = QFrame = QSizePolicy = QDrag = None
    _HAS_QT = False

class UserWarning:
    """Centralized warning system for user-friendly error messages."""
    
    @staticmethod
    def warn(title: str, message: str, severity: str = "WARNING") -> None:
        """Print a formatted warning message to the user."""
        print(f"\n{'=' * 70}")
        print(f"[{severity}] {title}")
        print(f"{'-' * 70}")
        print(message)
        print(f"{'=' * 70}\n")
    
    @staticmethod
    def warn_insufficient_ram(available_mb: int, required_mb: int) -> None:
        """Warn when there's not enough RAM for the model context."""
        message = f"""
Not enough system RAM available for optimal performance.
  • Available: {available_mb} MB
  • Recommended: {required_mb} MB
  • Shortfall: {required_mb - available_mb} MB

⚠️  Atlas will use a smaller context window to avoid crashes.
    This may reduce the quality of responses.

Solutions:
  • Close other applications to free up memory
  • Reduce background processes
  • Load a smaller model
  • Set ATLASAI_CTX_SIZE environment variable to override
"""
        UserWarning.warn("Insufficient RAM", message.strip(), "WARNING")
    
    @staticmethod
    def warn_model_too_large(model_size_mb: int, available_vram_mb: int) -> None:
        """Warn when model file is larger than available VRAM."""
        message = f"""
The model file is larger than available GPU memory.
  • Model size: {model_size_mb} MB
  • GPU VRAM available: {available_vram_mb} MB
  • Shortfall: {model_size_mb - available_vram_mb} MB

⚠️  The model will be loaded to CPU, which is MUCH slower.
    GPU acceleration will not be available.

Solutions:
  • Use a smaller/quantized model (e.g., Q4 instead of Q8)
  • Upgrade GPU or use a machine with more VRAM
  • Free up GPU memory (close other GPU apps)
"""
        UserWarning.warn("Model Larger Than GPU Memory", message.strip(), "WARNING")
    
    @staticmethod
    def warn_gpu_unavailable() -> None:
        """Warn when GPU is not available for acceleration."""
        message = """
GPU acceleration is not available or disabled.
  • Models will run on CPU (slow)
  • NVIDIA GPU detected but CUDA not working
  • Set ATLASAI_GPU_LAYERS=0 to use CPU intentionally

To enable GPU acceleration:
  • Install CUDA Toolkit matching your GPU
  • Install cuDNN libraries
  • Reinstall llama-cpp-python with: pip install --force-reinstall llama-cpp-python
  • Set CUDA_PATH environment variable
"""
        UserWarning.warn("GPU Acceleration Disabled", message.strip(), "WARNING")
    
    @staticmethod
    def warn_embedding_model_failed() -> None:
        """Warn when embedding model fails to load."""
        message = """
Failed to load the sentence transformer model.
  • Falling back to simple embedding (reduced quality)
  • Memory retrieval will be less accurate

This is usually due to:
  • First time downloading the model (will retry)
  • Internet connection issues
  • Low disk space for cache

Atlas will continue working with degraded memory quality.
"""
        UserWarning.warn("Embedding Model Load Failed", message.strip(), "WARNING")
    
    @staticmethod
    def warn_no_disk_space() -> None:
        """Warn when there's not enough disk space."""
        message = """
Low disk space detected.
  • Memory and chat history may fail to save
  • Model file may be incomplete

Free up disk space:
  • Delete old downloads or caches
  • Clear temporary files
  • Remove unused models from ~/Documents/Ai_Models/
"""
        UserWarning.warn("Low Disk Space", message.strip(), "WARNING")
    
    @staticmethod
    def warn_model_not_found(model_path: str) -> None:
        """Warn when model file is not found."""
        message = f"""
Model file not found: {model_path}

Possible solutions:
  • Check the path is correct
  • Verify the file exists
  • Download the model from HuggingFace
  • Place GGUF models in: ~/Documents/Ai_Models/
  • Use: !loadmodel <path> to load a different model

Current search directory: {MODEL_SEARCH_DIR}
Run !help for more commands.
"""
        UserWarning.warn("Model Not Found", message.strip(), "WARNING")
    
    @staticmethod
    def warn_llama_cpp_import_failed() -> None:
        """Warn when llama-cpp-python is not installed."""
        message = """
llama-cpp-python is not installed.

Install it with:
  pip install llama-cpp-python

Or for GPU support:
  # NVIDIA CUDA
  pip install llama-cpp-python[cuda]
  
  # AMD ROCm  
  pip install llama-cpp-python[rocm]
  
  # Apple Metal
  pip install llama-cpp-python[metal]

After installing, restart Atlas.
"""
        UserWarning.warn("Missing llama-cpp-python", message.strip(), "ERROR")


# Prefiring the GPU can help reduce latency on the first query, so we do a quick check here to see if we can use it and set the default number of layers accordingly.
# Prefiring n_ctx window to set automatically based on available VRAM or system RAM, with a safety buffer to avoid OOM crashes.
# This is a best-effort approach and may not be perfect, but it should help optimize the default settings for most users without requiring manual configuration.
def _auto_detect_gpu_layers() -> int:
    env_layers = os.environ.get("ATLASAI_GPU_LAYERS")
    if env_layers:
        try:
            return max(0, int(env_layers))
        except ValueError:
            pass
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        free_mb = int(result.stdout.strip())
        layers = max(0, (free_mb - 512) // 150)
        return min(layers, 99)
    except Exception:
        pass
    return 0

def _auto_detect_context_size(safety_buffer_mb: int = 512) -> int:
    try:
        import psutil
        free_mb = psutil.virtual_memory().available // (1024*1024)
        print(f"[Atlas] Available RAM: {free_mb}MB")
    except Exception as e:
        print(f"[Atlas] psutil check failed: {e}")
    env_ctx = os.environ.get("ATLASAI_CTX_SIZE")
    if env_ctx:
        try:
            return max(512, int(env_ctx))
        except ValueError:
            pass
    # Try VRAM first (GPU path)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        total_mb, free_mb = (int(x.strip()) for x in result.stdout.strip().split(","))
        # Reserve space for model weights (use total - free as model size estimate)
        model_mb = total_mb - free_mb
        available_mb = free_mb - safety_buffer_mb
        if available_mb > 0:
            #~0.125MB per token per layer for Q4, 28-32 layers typical
            n_layers = 32
            mb_per_token = (n_layers * 2 * 128) / 1024  # key + value, head_dim=128
            max_tokens = int(available_mb / mb_per_token)
            result_ctx = max(4096, min(max_tokens, 32768))
            if available_mb < 2000:  # Less than 2GB free
                UserWarning.warn_insufficient_ram(available_mb, 4000)
            return result_ctx
    except Exception:
        pass
    # Fall back to system RAM (CPU path) - much more conservative
    try:
        import psutil
        free_mb = psutil.virtual_memory().available // (1024 * 1024)
        available_mb = free_mb - safety_buffer_mb
        if available_mb > 0:
            max_tokens = int((available_mb / 0.5) * 1024)
            print(f"[Atlas] Calculated context size: {max_tokens} tokens")
            result_ctx = max(4096, min(max_tokens, 32768))  # cap at 32k for CPU
            # Warn if available RAM is less than 4GB
            if free_mb < 4096:
                UserWarning.warn_insufficient_ram(free_mb, 8192)
            return result_ctx
    except Exception:
        pass
    print("[Atlas] Using default context size (16384 tokens)")
    return 16384

MEMORY_DIR = os.path.join(pathlib.Path.home(), ".AtlasAI")
CHAT_LOG_DIR = os.path.join(MEMORY_DIR, "chats")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.jsonl")
CHAT_LOG_FILE = os.path.join(MEMORY_DIR, "chat_history.jsonl")
WORKSPACE_CONFIG_FILE = os.path.join(MEMORY_DIR, "workspace.json")
MODEL_SEARCH_DIR = os.path.expanduser("~/Documents/Ai_Models/")
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
SEARXNG_URL = os.environ.get("SEARXNG_URL")
SEARXNG_API_KEY = os.environ.get("SEARXNG_API_KEY")
DUCKDUCKGO_API = "https://duckduckgo.com/"
WEBSEARCH_TIMEOUT = 15
HALF_LIFE_SECONDS = 60 * 60 * 24 * 7
DEFAULT_GPU_LAYERS = _auto_detect_gpu_layers()
ENABLE_AUTO_MEMORY = os.environ.get("ATLASAI_AUTO_MEMORY", "1") == "1"
ENGRAM_TYPE_WEIGHTS = {
    "preference": 1.5,
    "manual": 1.4,
    "insight": 1.6,
    "web": 1.2,
    "fact": 1.0,
    "event": 1.1,
    "recent": 0.9,
}


def load_markdown_file(filename: str) -> str:
    search_paths = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except Exception:
                return ""
    return ""


def _check_system_resources() -> Dict[str, Any]:
    """Check available system resources and return a dict of resource info."""
    resources = {
        "ram_free_mb": 0,
        "disk_free_mb": 0,
        "vram_free_mb": 0,
        "has_gpu": False,
        "warnings": []
    }
    
    # Check RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        resources["ram_free_mb"] = vm.available // (1024 * 1024)
        if resources["ram_free_mb"] < 2048:  # Less than 2GB
            resources["warnings"].append("Low RAM (less than 2GB free)")
    except Exception:
        pass
    
    # Check disk space
    try:
        import shutil
        home_dir = pathlib.Path.home()
        stat = shutil.disk_usage(home_dir)
        resources["disk_free_mb"] = stat.free // (1024 * 1024)
        if resources["disk_free_mb"] < 2048:  # Less than 2GB
            resources["warnings"].append("Low disk space (less than 2GB free)")
    except Exception:
        pass
    
    # Check GPU/VRAM
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            resources["vram_free_mb"] = int(result.stdout.strip())
            resources["has_gpu"] = True
    except Exception:
        pass
    
    return resources


# new system prompt taking from a file instead of hardcoding it here, with a fallback to the old prompt if the file is not found or cannot be read.
SYSTEM_PROMPT = load_markdown_file("system_prompt.md")
# Printing the if it used the default prompt instead of the file-based one, to make it clear to the user what is being used.
if not SYSTEM_PROMPT:
    print("System prompt not found. Please create a file named 'system_prompt.md' in the same directory as Atlas.py with your desired prompt content.")
else:
    print("Loaded system prompt from 'system_prompt.md'.")
PROMPT_TEMPLATE = (
    "{system}\n\n"
    "MEMORY_FILE_SNAPSHOT (authoritative):\n{memory_section}\n\n"
    "RELEVANT_MEMORY_HITS:\n{context}\n\n"
    "{recent_section}"
    "{instructions_section}"
    "{tools_section}"
    "{workspace_section}"
    "User request:\n{user}\n\n"
    "{web_section}"
    "Assistant:\n"
)

READONLY_COMMANDS = ["!quit", "!exit", "!help", "!memory", "!clear"]
DEFAULT_INSTRUCTIONS_FILENAME = "instructions.md"
DEFAULT_TOOLS_FILENAME = "tools.md"
MAX_PROMPT_MEMORY_ENTRIES = 200
MAX_PROMPT_MEMORY_CHARS = 12000


def find_gguf_models(search_dir: str = MODEL_SEARCH_DIR) -> List[str]:
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(
            f"Model folder not found: {search_dir}. Place your GGUF model there or pass a specific path via --model."
        )

    results: List[str] = []
    for root, _, files in os.walk(search_dir):
        for name in files:
            if name.endswith(".gguf"):
                results.append(os.path.join(root, name))
    results.sort()
    return results

# "RAG brain" memory which stands for Retrieval-Augmented Generation, where we store facts, preferences, and other information in a vector database (in this case, a simple in-memory store with optional sentence transformer embeddings) and retrieve relevant pieces of information to include in the prompt for the language model. This allows the assistant to have a persistent memory that can be accessed and updated over time, improving its ability to provide accurate and contextually relevant responses.
class MemoryStore:
    def __init__(self, path: str, embed_model_name: str = EMBED_MODEL):
        self.path = path
        self.embed_model_name = embed_model_name
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.use_fallback = False
        self.model = None
        self.last_error = ""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._initialize_embedder()
        self.load()

    def _initialize_embedder(self) -> None:
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.embed_model_name)
            except Exception as e:
                self.model = None
                self.use_fallback = True
                self.last_error = str(e)
                UserWarning.warn_embedding_model_failed()
        else:
            self.use_fallback = True

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        if self.use_fallback or self.model is None:
            return np.vstack([simple_embedding(t) for t in texts]).astype("float32")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _build_embeddings(self) -> None:
        if not self.entries:
            self.embeddings = None
            return
        texts = [entry["text"] for entry in self.entries]
        self.embeddings = self._embed_text(texts)

    def add(self, text: str, tag: str = "fact", weight: float = 1.0, source: str = "") -> None:
        now = time.time()
        entry = {
            "text": text.strip(),
            "tag": tag,
            "weight": float(weight),
            "source": source,
            "timestamp": now,
        }
        self.entries.append(entry)
    
        # Only embed the new entry and append instead of rebuilding everything
        new_emb = self._embed_text([text.strip()])
        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])
        
        self.save()

    def search(self, query: str, top_k: int = 4) -> List[str]:
        if not self.entries:
            return []
        query_emb = self._embed_text([query])[0]
        if self.embeddings is None:
            return []
        scores = np.dot(self.embeddings, query_emb)
        now = time.time()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, sim in enumerate(scores.tolist()):
            entry = self.entries[i]
            age = now - entry.get("timestamp", now)
            decay = math.exp(-age / HALF_LIFE_SECONDS)
            type_weight = ENGRAM_TYPE_WEIGHTS.get(entry.get("tag", "fact"), 1.0)
            weight = float(entry.get("weight", 1.0)) * type_weight
            score = float(sim) * weight * decay
            scored.append((score, entry))

        # Dynamic threshold — mean + fraction of std dev
        all_scores = [s for s, _ in scored]
        if all_scores:
            mean = np.mean(all_scores)
            std = np.std(all_scores)
            threshold = max(0.05, mean + 0.3 * std)
        else:
            threshold = 0.05

        scored = [(s, e) for s, e in scored if s > threshold]
        scored.sort(reverse=True, key=lambda item: item[0])
        return [entry["text"] for _, entry in scored[:top_k]]

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for entry in self.entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self) -> None:
        self.entries = []
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "text" in entry:
                        entry.setdefault("tag", "fact")
                        entry.setdefault("weight", 1.0)
                        entry.setdefault("source", "")
                        entry.setdefault("timestamp", time.time())
                        self.entries.append(entry)
                except json.JSONDecodeError:
                    continue
        self._build_embeddings()


def simple_embedding(text: str) -> np.ndarray:
    text = text.lower().strip()
    if not text:
        return np.zeros(EMBED_DIM, dtype="float32")
    freq: Dict[str, int] = {}
    for i in range(len(text) - 1):
        bg = text[i : i + 2]
        freq[bg] = freq.get(bg, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:EMBED_DIM]
    vec = np.zeros(EMBED_DIM, dtype="float32")
    for idx, (_, count) in enumerate(items):
        vec[idx] = float(count)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

class WorkspaceManager:
    """Lazy, on-demand filesystem access for a user-chosen root directory."""

    MAX_FILE_CHARS   = 8_000
    MAX_TREE_ENTRIES = 200
    TEXT_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
        ".html", ".htm", ".css", ".scss", ".json", ".jsonl", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf", ".env", ".sh", ".bash", ".zsh",
        ".fish", ".rs", ".go", ".c", ".cpp", ".h", ".hpp", ".java", ".kt",
        ".rb", ".php", ".cs", ".swift", ".lua", ".r", ".sql", ".xml",
        ".csv", ".log", ".rst", ".tex", ".makefile", ".dockerfile", "",
    }

    def __init__(self, config_path: str = WORKSPACE_CONFIG_FILE):
        self.config_path = config_path
        self.root: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            root = data.get("root", "")
            if root and os.path.isdir(root):
                self.root = root
        except Exception:
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as fh:
                json.dump({"root": self.root or ""}, fh)
        except Exception:
            pass

    def set_root(self, path: str) -> str:
        path = os.path.expanduser(path.strip())
        if not os.path.isdir(path):
            return f"✗ Not a directory: {path}"
        self.root = path
        self._save()
        return f"✓ Workspace set to: {path}"

    def clear(self) -> str:
        self.root = None
        self._save()
        return "Workspace cleared."

    def status(self) -> str:
        if self.root:
            return f"Workspace root: {self.root}"
        return "No workspace set. Use !workspace <path> to set one."

    def tree(self, max_entries: int = MAX_TREE_ENTRIES) -> str:
        if not self.root:
            return "No workspace set."
        lines: List[str] = [f"Workspace: {self.root}"]
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]
            rel_dir = os.path.relpath(dirpath, self.root)
            depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
            indent = "  " * depth
            folder_name = os.path.basename(dirpath) if rel_dir != "." else "."
            lines.append(f"{indent}📁 {folder_name}/")
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.TEXT_EXTENSIONS:
                    continue
                lines.append(f"{indent}  {fname}")
                count += 1
                if count >= max_entries:
                    lines.append(f"  … (truncated at {max_entries} entries)")
                    return "\n".join(lines)
        return "\n".join(lines)

    def read_file(self, rel_or_abs: str) -> Tuple[str, str]:
        if not self.root:
            return "", "No workspace set."
        path = rel_or_abs.strip()
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        try:
            resolved = os.path.realpath(path)
            root_resolved = os.path.realpath(self.root)
            if not resolved.startswith(root_resolved + os.sep) and resolved != root_resolved:
                return "", "Access denied: path is outside workspace root."
        except Exception:
            return "", "Could not resolve path."
        if not os.path.isfile(resolved):
            return "", f"File not found: {resolved}"
        ext = os.path.splitext(resolved)[1].lower()
        if ext not in self.TEXT_EXTENSIONS:
            return "", f"Binary file skipped: {os.path.basename(resolved)}"
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(self.MAX_FILE_CHARS)
            truncated = os.path.getsize(resolved) > self.MAX_FILE_CHARS
            note = f"\n… [truncated at {self.MAX_FILE_CHARS} chars]" if truncated else ""
            return content + note, ""
        except Exception as exc:
            return "", f"Read error: {exc}"

    def find_relevant_files(self, query: str, max_files: int = 3) -> List[Tuple[str, str]]:
        if not self.root:
            return []
        query_lower = query.lower()
        terms = [re.sub(r"[^\w]", "", w) for w in query_lower.split() if len(w) > 3]
        if not terms:
            return []
        matches: List[Tuple[int, str]] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.TEXT_EXTENSIONS:
                    continue
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, self.root)
                combined = (fname + " " + rel_path).lower()
                score = sum(1 for t in terms if t in combined)
                if score > 0:
                    matches.append((score, abs_path))
        matches.sort(reverse=True, key=lambda x: x[0])
        results = []
        for _, abs_path in matches[:max_files]:
            content, err = self.read_file(abs_path)
            if content:
                rel = os.path.relpath(abs_path, self.root)
                results.append((rel, content))
        return results

    def prompt_section(self, query: str) -> str:
        if not self.root:
            return ""
        relevant = self.find_relevant_files(query)
        if not relevant:
            return f"WORKSPACE ROOT: {self.root}\n(No files matched query — use !workspace read <file> to load one.)\n\n"
        parts = [f"WORKSPACE ROOT: {self.root}"]
        for rel_path, content in relevant:
            parts.append(f"\n--- FILE: {rel_path} ---\n{content}\n--- END FILE ---")
        parts.append("")
        return "\n".join(parts) + "\n"



def extract_json_text(text: str) -> str:
    """
    Extract the first JSON object or array from the model output.
    Uses a stack to find the matching closing bracket in O(n).
    """
    text = text.strip()
    if not text:
        return ""

    openers = {"{": "}", "[": "]"}
    closers = set(openers.values())

    for start, ch in enumerate(text):
        if ch not in openers:
            continue
        stack = []
        in_string = False
        escape = False
        for end in range(start, len(text)):
            c = text[end]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in openers:
                stack.append(openers[c])
            elif c in closers:
                if not stack or stack[-1] != c:
                    break
                stack.pop()
                if not stack:
                    candidate = text[start:end + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    return ""

def safe_eval_math(expression: str) -> str:
    try:
        expression = expression.strip()
        if not expression:
            return ""
        if re.search(r"[^0-9\.\+\-\*/\(\) \t]", expression):
            return ""
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return ""

def is_math_query(text: str) -> bool:
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", text):
        return True
    return False


def format_conversation(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for item in history[-20:]:
        prefix = "User" if item.get("role") == "user" else "Assistant"
        message = item.get("message", "")
        lines.append(f"{prefix}: {message}")
    return "\n".join(lines)


def _searxng_search(query: str, max_results: int = 4) -> Optional[Dict[str, Any]]:
    """Search using custom searXNG instance. Returns None if failed."""
    if not SEARXNG_URL:
        return None
    
    try:
        params = {
            "q": query,
            "format": "json",
        }
        headers = {}
        if SEARXNG_API_KEY:
            headers["Authorization"] = f"Bearer {SEARXNG_API_KEY}"
        
        response = requests.get(SEARXNG_URL, params=params, headers=headers, timeout=WEBSEARCH_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    
    results = data.get("results", [])
    sources: List[Dict[str, str]] = []
    summary = ""
    
    for result in results[:max_results]:
        if "title" in result and "url" in result:
            sources.append({
                "title": result.get("title", ""),
                "url": result.get("url", "")
            })
            if not summary and "content" in result:
                summary = result.get("content", "")
    
    if not summary and sources:
        summary = sources[0].get("title", "")
    
    return {"query": query, "summary": summary.strip(), "sources": sources}


def duckduckgo_search(query: str, max_results: int = 4) -> Dict[str, Any]:
    """Search using searXNG if available, fallback to DuckDuckGo."""
    # Try searXNG first if configured
    searxng_result = _searxng_search(query, max_results)
    if searxng_result is not None:
        return searxng_result
    
    # Fallback to DuckDuckGo
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "t": "atlasai",
        }
        response = requests.get(DUCKDUCKGO_API, params=params, timeout=WEBSEARCH_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {"query": query, "summary": "", "sources": [], "error": str(exc)}

    summary = data.get("AbstractText", "") or data.get("Heading", "")
    sources: List[Dict[str, str]] = []
    abstract_url = data.get("AbstractURL", "")
    if abstract_url:
        sources.append({"title": data.get("Heading", "DuckDuckGo"), "url": abstract_url})

    topics = data.get("RelatedTopics", [])
    if isinstance(topics, list):
        for topic in topics:
            if len(sources) >= max_results:
                break
            if isinstance(topic, dict):
                if topic.get("Text") and topic.get("FirstURL"):
                    sources.append({"title": topic.get("Text"), "url": topic.get("FirstURL")})
                elif topic.get("Topics"):
                    for sub in topic.get("Topics", []):
                        if len(sources) >= max_results:
                            break
                        if sub.get("Text") and sub.get("FirstURL"):
                            sources.append({"title": sub.get("Text"), "url": sub.get("FirstURL")})

    if not summary and sources:
        summary = sources[0].get("title", "")

    if not summary and not sources:
        summary = "No instant answer available from DuckDuckGo."

    return {"query": query, "summary": summary.strip(), "sources": sources}


class AtlasAI:
    def __init__(self, model_path: Optional[str] = None, memory_path: str = MEMORY_FILE):
        self.model_path = model_path
        self.memory = MemoryStore(memory_path)
        self.workspace = WorkspaceManager()
        self.history: List[Dict[str, str]] = []
        self.chat_filename: Optional[str] = None
        self.last_prompt = ""
        self.last_raw_response = ""
        self.gpu_layers = int(os.environ.get("ATLASAI_GPU_LAYERS", DEFAULT_GPU_LAYERS))
        self.auto_save_memory = ENABLE_AUTO_MEMORY
        self.llm: Optional[Llama] = None
        # --- prompt_counter: triggers synthesize_new_memory_node every 5 active cycles ---
        self._prompt_counter: int = 0
        self._MEMORY_SYNTHESIS_INTERVAL: int = 5
        if model_path:
            self.llm = self._load_model(model_path)
        self._print_startup_info()

    def _load_model(self, model_path: str) -> Llama:
        if not model_path:
            raise ValueError("Model path is required to load a model.")
        if Llama is None:
            UserWarning.warn_llama_cpp_import_failed()
            raise ImportError("Atlas requires llama-cpp-python. Install it with: pip install llama-cpp-python")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            UserWarning.warn_model_not_found(model_path)
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Get model file size
        try:
            model_size_mb = os.path.getsize(model_path) // (1024 * 1024)
            print(f"[Atlas] Model file size: {model_size_mb} MB")
        except Exception as e:
            print(f"[Atlas] Could not determine model size: {e}")
            model_size_mb = 0
        
        # Check available VRAM and warn if model is larger
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                free_vram_mb = int(result.stdout.strip())
                if model_size_mb > free_vram_mb:
                    UserWarning.warn_model_too_large(model_size_mb, free_vram_mb)
        except Exception:
            pass  # nvidia-smi not available or failed
        
        # Check available system RAM
        try:
            import psutil
            available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
            if model_size_mb > available_ram_mb:
                UserWarning.warn_model_too_large(model_size_mb, available_ram_mb)
        except Exception:
            pass
        
        os.environ["LLAMA_CUBLAS"] = "1"
        os.environ["GGML_CUBLAS"] = "1"
        os.environ["GGML_CUDA_FORCE_CUBLAS"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 1)

        self.gpu_layers = int(os.environ.get("ATLASAI_GPU_LAYERS", DEFAULT_GPU_LAYERS))
        
        # Warn if GPU acceleration is disabled
        if self.gpu_layers == 0 and DEFAULT_GPU_LAYERS == 0:
            print("[Atlas] GPU acceleration is not enabled. Models will run on CPU (slower).")
        
        model_kwargs = {
            "model_path": model_path,
            "n_ctx": _auto_detect_context_size(),
            "main_gpu": 0,
            "n_gpu_layers": self.gpu_layers,
            "n_threads": os.cpu_count() or 1,
            "use_mlock": False,
            "use_mmap": True,
            "top_k": 40,
            "top_p": 0.92,
            "temperature": 0.1,
            "repeat_penalty": 1.2,
        }

        try:
            print("[Atlas] Loading model... (this may take a moment)")
            model = Llama(**model_kwargs)
            print("[Atlas] Model loaded successfully!")
            return model
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                UserWarning.warn_gpu_unavailable()
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                UserWarning.warn_insufficient_ram(0, model_size_mb)
            raise

    def _sanitize_chat_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        cleaned = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+){0,2}", cleaned):
            return None
        return cleaned.lower().replace(" ", "_")

    def _derive_chat_name(self) -> str:
        user_messages = [entry["message"] for entry in self.history if entry["role"] == "user"]
        if not user_messages:
            return "last_session"

        candidate = next((msg for msg in user_messages if not msg.strip().startswith("!")), user_messages[0])
        candidate = candidate.strip().lower()
        candidate = re.sub(r"[^a-z0-9 ]+", "", candidate)
        words = [word for word in candidate.split() if word]
        if not words:
            return "last_session"

        return "_".join(words[:4])

    def _ensure_chat_filename(self) -> str:
        if self.chat_filename:
            return self.chat_filename

        base_name = self._derive_chat_name()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.chat_filename = f"{base_name}_{timestamp}"
        return self.chat_filename

    def _chat_history_filepath(self, sanitized_name: str) -> str:
        return os.path.join(CHAT_LOG_DIR, f"{sanitized_name}.jsonl")

    def save_chat_history(self, name: Optional[str] = None) -> str:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)
        if name:
            sanitized = self._sanitize_chat_name(name)
            if not sanitized:
                return "Chat name must be 1-3 words containing only letters and numbers."
            filepath = self._chat_history_filepath(sanitized)
            self.chat_filename = sanitized
        else:
            filename = self._ensure_chat_filename()
            filepath = self._chat_history_filepath(filename)

        with open(filepath, "w", encoding="utf-8") as fh:
            for entry in self.history:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if name:
            return f"Chat saved as '{name}' in {filepath}."
        return f"Chat saved to {filepath}."

    def load_chat_history(self, name: str) -> str:
        sanitized = self._sanitize_chat_name(name)
        if not sanitized:
            return "Chat name must be 1-3 words containing only letters and numbers."
        filepath = self._chat_history_filepath(sanitized)
        if not os.path.exists(filepath):
            return f"Saved chat '{name}' not found."

        loaded: List[Dict[str, str]] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "role" in entry and "message" in entry:
                        loaded.append(entry)
                except json.JSONDecodeError:
                    continue

        self.history = loaded
        self.chat_filename = sanitized
        return f"Loaded chat '{name}' with {len(self.history)} messages."

    def load_chat_history_file(self, path: str) -> str:
        if not os.path.exists(path):
            return f"Chat file not found: {path}"

        loaded: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "role" in entry and "message" in entry:
                        loaded.append(entry)
                except json.JSONDecodeError:
                    continue

        self.history = loaded
        self.chat_filename = os.path.splitext(os.path.basename(path))[0]
        return f"Loaded chat file {os.path.basename(path)} with {len(self.history)} messages."

    def list_chat_history(self) -> str:
        if not os.path.isdir(CHAT_LOG_DIR):
            return "No saved chats found."

        files = [f for f in os.listdir(CHAT_LOG_DIR) if f.endswith(".jsonl")]
        if not files:
            return "No saved chats found."

        lines = ["Saved chats:"]
        for filename in sorted(files):
            name = os.path.splitext(filename)[0].replace("_", " ")
            lines.append(f"- {name}")
        return "\n".join(lines)

    def load_model(self, model_path: str) -> str:
        if not model_path:
            return "Usage: !loadmodel /path/to/model.gguf"
        if os.path.isdir(model_path):
            models = find_gguf_models(model_path)
            if not models:
                return f"No GGUF models found in directory: {model_path}"
            model_path = models[0]

        if not os.path.exists(model_path):
            UserWarning.warn_model_not_found(model_path)
            return f"Model path not found: {model_path}"

        if self.llm is not None:
            self.llm = None
            import gc; gc.collect()

        self.model_path = model_path
        try:
            self.llm = self._load_model(model_path)
            return f"✓ Model loaded successfully: {os.path.basename(self.model_path)}"
        except FileNotFoundError as e:
            return f"✗ Model not found: {e}"
        except ImportError as e:
            return f"✗ Missing dependency: {e}"
        except RuntimeError as e:
            if "CUDA" in str(e):
                return "✗ GPU error. Model will run on CPU. This will be very slow."
            if "out of memory" in str(e).lower():
                return f"✗ Out of memory error. Try a smaller model or close other applications."
            return f"✗ Runtime error: {e}"
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if "oom" in error_msg.lower() or "memory" in error_msg.lower():
                return f"✗ Out of memory. The model is too large for available RAM/VRAM."
            return f"✗ Failed to load model ({error_type}): {error_msg}"

    def handle_command(self, user: str) -> Optional[str]:
        lowered = user.lower().strip()
        if lowered == "!help":
            return self._print_help()
        if lowered == "!memory":
            return self.show_memory()
        if lowered == "!clear":
            return self.clear_memory()
        if lowered.startswith("!remember "):
            note = user[len("!remember "):].strip()
            self.memory.add(note, tag="manual")
            return "Saved to memory."
        if lowered.startswith("!savechat "):
            name = user.split(maxsplit=1)[1].strip() if len(user.split(maxsplit=1)) > 1 else ""
            if not name:
                return "Usage: !savechat <name> (1-3 words)"
            return self.save_chat_history(name)
        if lowered == "!savechat":
            return "Usage: !savechat <name> (1-3 words)"
        if lowered == "!listchats":
            return self.list_chat_history()
        if lowered.startswith("!loadchat "):
            name = user.split(maxsplit=1)[1].strip() if len(user.split(maxsplit=1)) > 1 else ""
            if not name:
                return "Usage: !loadchat <name>"
            return self.load_chat_history(name)
        if lowered == "!loadchat":
            return "Usage: !loadchat <name>"
        if lowered == "!chatlog":
            return f"Chat log saved at: {CHAT_LOG_DIR}"
        if lowered.startswith("!loadmodel ") or lowered.startswith("!model "):
            model_path = user.split(maxsplit=1)[1].strip() if len(user.split(maxsplit=1)) > 1 else ""
            return self.load_model(model_path)
        if lowered in ("!loadmodel", "!model"):
            return "Usage: !loadmodel /path/to/model.gguf"
        # Workspace commands
        if lowered == "!workspace":
            return self.workspace.status()
        if lowered.startswith("!workspace set "):
            path = user[len("!workspace set "):].strip()
            return self.workspace.set_root(path)
        if lowered.startswith("!workspace ") and not lowered.startswith("!workspace read") and not lowered.startswith("!workspace tree") and not lowered.startswith("!workspace clear") and not lowered.startswith("!workspace set"):
            # bare: !workspace /some/path
            path = user[len("!workspace "):].strip()
            if path and not path.startswith("!"):
                return self.workspace.set_root(path)
        if lowered == "!workspace tree":
            return self.workspace.tree()
        if lowered == "!workspace clear":
            return self.workspace.clear()
        if lowered.startswith("!workspace read "):
            rel = user[len("!workspace read "):].strip()
            content, err = self.workspace.read_file(rel)
            if err:
                return f"✗ {err}"
            return f"--- {rel} ---\n{content}\n--- end ---"
        return None

    def _print_startup_info(self) -> None:
        print("AtlasAI is ready.")
        model_name = self.model_path if self.model_path else "No model loaded"
        print(f"Model: {model_name}")
        print(f"Memory: {self.memory.path}")
        print(f"Memory entries: {len(self.memory.entries)}")
        if self.memory.use_fallback:
            print("Embedding: fallback mode (reduced quality)")
        else:
            print(f"Embedding: {self.memory.embed_model_name}")
        if self.gpu_layers > 0:
            print(f"GPU acceleration: enabled ({self.gpu_layers} layer(s) on GPU)")
        else:
            print("GPU acceleration: disabled (CPU mode will be slower)")
        print(f"Auto memory saving: {'enabled' if self.auto_save_memory else 'disabled'}")
        
        # Check system resources and display warnings
        resources = _check_system_resources()
        if resources["warnings"]:
            print("\n⚠️  Resource Warnings:")
            for warning in resources["warnings"]:
                print(f"   • {warning}")
        
        if not _HAS_QT:
            print("\nℹ️  GUI support unavailable. Install PySide6 or run with --cli for console mode.")
        print("---")
        print("Type '!help' for commands. Start typing your question.")
        print("---")

    def build_prompt(self, user: str, retrieved: List[str], web_summary: str = "", web_sources: str = "") -> str:
        context = "\n".join(retrieved) if retrieved else "No relevant memory found."
        memory_section = self._memory_snapshot_for_prompt()
        recent_context = format_conversation(self.history)
        recent_section = "Recent conversation:\n" + recent_context + "\n\n" if recent_context else ""
        instructions = load_markdown_file(DEFAULT_INSTRUCTIONS_FILENAME)
        tools = load_markdown_file(DEFAULT_TOOLS_FILENAME)
        instructions_section = "Instructions file:\n" + instructions + "\n\n" if instructions else ""
        tools_section = "Tools file:\n" + tools + "\n\n" if tools else ""
        workspace_section = self.workspace.prompt_section(user)
        web_section = ""
        if web_summary or web_sources:
            web_section = "Web search summary:\n" + web_summary.strip() + "\n\n"
            if web_sources:
                web_section += "Web sources:\n" + web_sources.strip() + "\n\n"
        return PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            memory_section=memory_section,
            context=context,
            user=user,
            web_section=web_section,
            recent_section=recent_section,
            instructions_section=instructions_section,
            tools_section=tools_section,
            workspace_section=workspace_section,
        )

    def _memory_snapshot_for_prompt(self) -> str:
        if not self.memory.entries:
            return "Memory is empty."

        lines: List[str] = []
        for entry in self.memory.entries[-MAX_PROMPT_MEMORY_ENTRIES:]:
            tag = str(entry.get("tag", "fact"))
            weight = float(entry.get("weight", 1.0))
            text = str(entry.get("text", "")).strip().replace("\n", " ")
            if not text:
                continue
            lines.append(f"- [{tag} w={weight:.2f}] {text}")

        if not lines:
            return "Memory is empty."

        snapshot = "\n".join(lines)
        if len(snapshot) > MAX_PROMPT_MEMORY_CHARS:
            snapshot = snapshot[-MAX_PROMPT_MEMORY_CHARS:]
            first_newline = snapshot.find("\n")
            if first_newline != -1:
                snapshot = snapshot[first_newline + 1 :]
            snapshot = "[Truncated memory snapshot]\n" + snapshot
        return snapshot

    def _should_search(self, user: str) -> bool:
        """Return True if the query is likely to benefit from a web search."""
        lowered = user.lower().strip()
        # Explicit search command
        if lowered.startswith("!search "):
            return True
        # Commands never need searching
        if lowered.startswith("!"):
            return False
        # Recency / news signals
        recency_signals = [
            "latest", "newest", "recent", "right now", "currently", "today",
            "this week", "this month", "this year", "in 2024", "in 2025", "in 2026",
            "news", "update", "current", "now", "just released", "just announced",
            "happening", "live", "breaking",
        ]
        if any(signal in lowered for signal in recency_signals):
            return True
        # Question words about facts that change over time
        factual_patterns = [
            r"\bwho is\b", r"\bwho are\b", r"\bwhat is\b", r"\bwhat are\b",
            r"\bwhere is\b", r"\bwhen (is|was|did|will)\b", r"\bhow (much|many|long|old)\b",
            r"\bprice of\b", r"\bcost of\b", r"\bweather\b", r"\bstock\b",
        ]
        if any(re.search(p, lowered) for p in factual_patterns):
            return True
        return False

    def _prepare_query(self, user: str) -> Tuple[str, str, str]:
        web_summary = ""
        web_sources = ""
        query = user
        if user.lower().startswith("!search "):
            query = user[len("!search "):].strip()
        if self._should_search(user):
            search_data = duckduckgo_search(query)
            web_summary = search_data.get("summary", "")
            sources = search_data.get("sources", [])
            if sources:
                web_sources = "\n".join([f"{src.get('title','')} - {src.get('url','')}" for src in sources])
        return query, web_summary, web_sources

    def _run_query(self, user: str, token_callback=None) -> str:
        if self.llm is None:
            raise RuntimeError("No model loaded. Load a GGUF model before running queries.")
        
        try:
            query, web_summary, web_sources = self._prepare_query(user)
            retrieved = self.memory.search(query)
            prompt = self.build_prompt(query, retrieved, web_summary=web_summary, web_sources=web_sources)
            self.last_prompt = prompt

            stream = token_callback is not None
            response = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.92,
                repeat_penalty=1.2,
                stop=["\nUser:", "\nAssistant:"],
                stream=stream,
            )

            if stream:
                collected = []
                for chunk in response:
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        collected.append(token)
                        token_callback("".join(collected))
                text = "".join(collected).strip()
            else:
                try:
                    if isinstance(response, dict):
                        choices = response.get("choices")
                        if isinstance(choices, list) and choices:
                            text = str(choices[0].get("text", "")).strip()
                        else:
                            text = str(response.get("text", "")).strip()
                    else:
                        text = str(response).strip()
                except Exception as exc:
                    text = f"[Error parsing model response: {exc}]"

            self.last_raw_response = text
            return text
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                raise RuntimeError("Out of memory or GPU error during model inference")
            raise
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}")

    def _split_answer_details(self, text: str) -> Tuple[str, str]:
        text = text.strip()
        markers = ["\nDetails:", "\nThoughts:", "\nReasoning:", "\nThought:", "\nDetail:"]
        for marker in markers:
            if marker in text:
                answer, details = text.split(marker, 1)
                return answer.strip(), details.strip()
        return text, ""

    def respond(self, user: str) -> str:
        if not user:
            return ""
        special = self._handle_special_cases(user)
        if special is not None:
            return special
        try:
            raw = self._run_query(user)
            answer, _ = self._split_answer_details(raw)
            answer = self._clean_output(answer)
            if self.auto_save_memory:
                try:
                    self._auto_save_memory(user, answer)
                except Exception:
                    pass
            # --- prompt_counter: synthesise a memory node every N cycles ---
            self._prompt_counter += 1
            if self._prompt_counter % self._MEMORY_SYNTHESIS_INTERVAL == 0:
                try:
                    self.synthesize_new_memory_node()
                except Exception:
                    pass
            return answer
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return "⚠️  Out of memory! The model query was too large. Try a simpler question or close other applications."
            return f"⚠️  Runtime error: {e}"
        except Exception as exc:
            error_msg = str(exc).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                return "⚠️  GPU error occurred. Falling back to CPU (slower)."
            if "model" in error_msg and "not" in error_msg:
                return "⚠️  No model loaded. Use !loadmodel to load a model first."
            return f"⚠️  Error generating response: {exc}"

    def respond_with_details(self, user: str, token_callback=None) -> Tuple[str, str]:
        if not user:
            return "", ""
        special = self._handle_special_cases(user)
        if special is not None:
            return special, ""
        try:
            raw = self._run_query(user, token_callback=token_callback)
            answer, details = self._split_answer_details(raw)
            answer = self._clean_output(answer)
            if self.auto_save_memory:
                try:
                    self._auto_save_memory(user, answer)
                except Exception:
                    pass
            # --- prompt_counter: synthesise a memory node every N cycles ---
            self._prompt_counter += 1
            if self._prompt_counter % self._MEMORY_SYNTHESIS_INTERVAL == 0:
                try:
                    self.synthesize_new_memory_node()
                except Exception:
                    pass
            return answer, details
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                msg = "⚠️  Out of memory! The model query was too large. Try a simpler question or close other applications."
                return msg, ""
            return f"⚠️  Runtime error: {e}", ""
        except Exception as exc:
            error_msg = str(exc).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                return "⚠️  GPU error occurred. Falling back to CPU (slower).", ""
            if "model" in error_msg and "not" in error_msg:
                return "⚠️  No model loaded. Use !loadmodel to load a model first.", ""
            return f"⚠️  Error generating response: {exc}", ""

    def _handle_special_cases(self, user: str) -> Optional[str]:
        lowered = user.lower().strip()
        memory_queries = [
            "what do you remember",
            "show memory",
            "what is in memory",
            "what's in memory",
            "memory entries",
            "saved memory",
            "recall memory",
        ]
        if any(phrase in lowered for phrase in memory_queries):
            return self.show_memory()

        # Intercept explicit save requests so they are guaranteed to persist
        # before the LLM generates any response.
        to_save = self._detect_save_intent(user)
        if to_save:
            tag = "manual"
            if any(w in lowered for w in ["prefer", "like", "love", "hate", "enjoy", "favorite"]):
                tag = "preference"
            elif any(w in lowered for w in ["i am", "i'm", "my name", "i work", "i live"]):
                tag = "fact"
            self.memory.add(to_save, tag=tag, weight=1.5, source="user_request")
            return f"Got it, I'll remember that: {to_save}"

        if is_math_query(user):
            expr_match = re.search(r"([0-9\.\s\+\-\*/\(\)]+)", user)
            if expr_match:
                calc = safe_eval_math(expr_match.group(1))
                if calc:
                    return f"Answer: {calc}"
        return None

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "I couldn't generate a response."
        text = re.sub(r"^\s*(assistant:|assistant\n|response:|answer:)\s*", "", text, flags=re.I)
        return text

    def _render_markdown_for_gui(self, text: str) -> str:
        return render_markdown_to_html(text)

    def _auto_save_memory(self, user: str, response: str, force_save: bool = False, tag: str = "fact", weight: float = 1.2) -> None:
        if self.llm is None:
            return

        if force_save:
            summarize_prompt = (
                "Summarize the following into a single clean memory entry, max 20 words. "
                "Return only the summary, nothing else.\n\n"
                f"User: {user}\nAssistant: {response}"
            )
            try:
                result = self.llm(summarize_prompt, max_tokens=60, temperature=0.0, stream=False)
                if isinstance(result, dict):
                    summary = result.get("choices", [{"text": ""}])[0].get("text", "").strip()
                else:
                    summary = str(result).strip()
                if summary:
                    self.memory.add(summary, tag=tag, weight=weight, source="user_request")
            except Exception:
                pass
            return

        decision_prompt = (
            "You are a memory curator for AtlasAI. Decide if the following exchange should be saved to long-term memory. "
            "Save only user preferences, stable personal data, recurring goals, important facts, project details, or useful web discoveries. "
            "Do not save casual chit-chat or one-off questions. "
            "Respond with a JSON object exactly like: {{\"save\": true|false, \"summary\": \"...\", \"tag\": \"preference|fact|web|event|manual\"}}."
            "\n\nRecent history:\n{recent}\n\nUser: {user}\nAssistant: {response}\n"
        )
        recent_context = format_conversation(self.history[-20:])
        prompt = decision_prompt.format(recent=recent_context, user=user, response=response)
        try:
            decision_response = self.llm(prompt, max_tokens=180, temperature=0.0, top_p=0.9, stream=False)
            if isinstance(decision_response, dict):
                text = decision_response.get("choices", [{"text": ""}])[0].get("text", "")
            else:
                text = str(decision_response)
            json_text = extract_json_text(text)
            decision = json.loads(json_text) if json_text else {}
        except Exception:
            decision = {}

        save = bool(decision.get("save", False))
        summary = str(decision.get("summary", "")).strip()
        tag = str(decision.get("tag", tag)).strip() or tag
        if save and summary:
            self.memory.add(summary, tag=tag, weight=weight, source="auto")

    def synthesize_new_memory_node(self) -> None:
        """Background synthesis: every N chat cycles, ask the LLM to distil a punchy
        long-term insight from recent history and commit it to the vector store with
        tag='insight' so it gets a slightly elevated engram weight."""
        if self.llm is None:
            return
        recent = format_conversation(self.history[-30:])
        synthesis_prompt = (
            "You are a memory archivist for AtlasAI. "
            "Read the conversation below and write ONE single punchy insight sentence "
            "(max 25 words) that captures a durable user trait, preference trend, or "
            "behavioural pattern worth remembering long-term. "
            "Return ONLY that sentence, nothing else.\n\n"
            f"Conversation:\n{recent}\n\nInsight:"
        )
        try:
            result = self.llm(synthesis_prompt, max_tokens=60, temperature=0.15, stream=False)
            if isinstance(result, dict):
                insight = result.get("choices", [{"text": ""}])[0].get("text", "").strip()
            else:
                insight = str(result).strip()
            # Strip any accidental leading labels like "Insight:" the model may echo back
            insight = re.sub(r"^(?:insight|summary)[:\s]+", "", insight, flags=re.I).strip()
            if insight and len(insight) > 8:
                self.memory.add(insight, tag="insight", weight=1.6, source="synthesis")
                print(f"[Atlas] Memory synthesis committed: {insight[:80]}")
        except Exception as exc:
            print(f"[Atlas] Memory synthesis skipped: {exc}")

    def _detect_save_intent(self, user: str) -> Optional[str]:
        """Return the text to save if the user explicitly asks Atlas to remember something."""
        lowered = user.lower().strip()
        explicit_patterns = [
            r"(?:please\s+)?(?:remember|save|note|store|keep track of)\s+(?:that\s+)?(?:my\s+)?(.+)",
            r"(?:i want you to|can you|could you)\s+(?:remember|save|note)\s+(?:that\s+)?(.+)",
            r"(?:make a note|make note)\s+(?:that\s+)?(.+)",
            r"(?:add to memory|save to memory|store in memory)\s*[:\-]?\s*(.+)",
        ]
        for pattern in explicit_patterns:
            m = re.search(pattern, lowered)
            if m:
                captured = m.group(1).strip().rstrip(".!?")
                if len(captured) >= 4:
                    return captured
        return None

    def add_memory_if_relevant(self, user: str, response: str) -> None:
        # Explicit save-intent: extract what to remember and persist it immediately.
        to_save = self._detect_save_intent(user)
        if to_save:
            self.memory.add(to_save, tag="manual", weight=1.5, source="user_request")
            return

        # Passive preference/identity signals.
        lowered = user.lower()
        triggers = ["i prefer", "i like", "my favorite", "i'm", "i am"]
        if any(trigger in lowered for trigger in triggers):
            self.memory.add(user.strip(), tag="preference", weight=1.4, source="passive")

    def show_memory(self) -> str:
        memory_error = getattr(self.memory, "last_error", "")
        if memory_error:
            return f"Memory warning: {memory_error}"
        if not self.memory.entries:
            return "Memory is empty."
        scored = []
        now = time.time()
        for entry in self.memory.entries:
            age = now - entry.get("timestamp", now)
            decay = math.exp(-age / HALF_LIFE_SECONDS)
            type_weight = ENGRAM_TYPE_WEIGHTS.get(entry.get("tag", "fact"), 1.0)
            score = float(entry.get("weight", 1.0)) * type_weight * decay
            scored.append((score, entry))
        scored.sort(reverse=True, key=lambda item: item[0])
        lines = []
        for score, entry in scored[:10]:
            age_hours = int((now - entry.get("timestamp", now)) / 3600)
            lines.append(
                f"- [{entry.get('tag','fact')} w={entry.get('weight',1.0):.2f}] {entry['text']} (age={age_hours}h, score={score:.3f})"
            )
        return "Memory:\n" + "\n".join(lines)

    def clear_memory(self) -> str:
        self.memory.entries = []
        self.memory.embeddings = None
        self.memory.save()
        return "Memory cleared."

    def run(self) -> None:
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break

            if not user:
                continue
            lower = user.lower()

            if lower in ("!quit", "!exit"):
                print("Goodbye.")
                break

            self.history.append({"role": "user", "message": user})
            command_response = self.handle_command(user)
            if command_response is not None:
                print(command_response)
                self.history.append({"role": "assistant", "message": command_response})
                self.save_chat_history()
                continue

            answer = self.respond(user)
            print(f"Atlas: {answer}\n")
            self.history.append({"role": "assistant", "message": answer})
            self.add_memory_if_relevant(user, answer)
            self.save_chat_history()
            self._current_stream_bubble = None

    def _print_help(self) -> str:
        return (
            "Commands:\n"
            "  !help              Show this command list\n"
            "  !memory            Show recent memory entries\n"
            "  !clear             Clear all saved memory\n"
            "  !remember X        Save a note to memory\n"
            "  !savechat X        Save the full chat history to disk under name X\n"
            "  !loadchat X        Load a saved chat by name\n"
            "  !listchats         List all saved chats\n"
            "  !chatlog           Show saved chat log location\n"
            "  !loadmodel X       Load a new GGUF model at runtime\n"
            "  !model X           Alias for !loadmodel\n"
            "  !workspace <path>  Set workspace root directory (persisted)\n"
            "  !workspace tree    Show file tree of current workspace\n"
            "  !workspace read X  Read a file from the workspace into chat\n"
            "  !workspace clear   Unset the current workspace\n"
            "  !exit              Quit the assistant\n"
        )

    def _render_markdown_for_gui(self, text: str) -> str:
        import html as htmllib

        # Pull out code blocks first so we don't mangle their contents
        code_blocks: list[str] = []
        def stash_code_block(m: re.Match) -> str:
            lang = m.group(1) or "plaintext"
            code = htmllib.escape(m.group(2).rstrip())
            block = (
                f"<pre style='background:#0d1117; color:#c9d1d9; padding:12px; "
                f"border-radius:6px; font-family:monospace; font-size:13px; "
                f"white-space:pre-wrap; border:1px solid #30363d; margin:8px 0;'>"
                f"<span style='color:#79c0ff; font-size:11px;'>{htmllib.escape(lang)}</span>\n{code}</pre>"
            )
            code_blocks.append(block)
            return f"\x00CODE{len(code_blocks) - 1}\x00"

        text = re.sub(r'```(\w+)?\n?(.*?)```', stash_code_block, text, flags=re.DOTALL)

        # Escape remaining HTML
        text = htmllib.escape(text)

        # Escaped markdown characters (e.g. \* \_ \~)
        text = re.sub(r'\\([*_~`#\-\[\]\\])', lambda m: f"\x01{ord(m.group(1))}\x01", text)

        # Inline code
        text = re.sub(
            r'`([^`]+)`',
            r"<code style='background:#1e293b; color:#60a5fa; padding:2px 6px; border-radius:3px; font-family:monospace; font-size:13px;'>\1</code>",
            text
        )

        # Bold + italic (*** or ___)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i style="color:#f0f6fc;">\1</i></b>', text)

        # Bold (** or __)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b style="color:#f0f6fc;">\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b style="color:#f0f6fc;">\1</b>', text)

        # Italic (* or _)
        text = re.sub(r'\*(.+?)\*', r'<i style="color:#cbd5e1;">\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i style="color:#cbd5e1;">\1</i>', text)

        # Strikethrough
        text = re.sub(r'~~(.+?)~~', r'<s style="color:#64748b;">\1</s>', text)

        # Headers (h3 before h2 before h1 to avoid greedy overlap)
        text = re.sub(r'^### (.+)$', r"<h3 style='color:#60a5fa; font-size:15px; font-weight:700; margin:12px 0 6px 0;'>\1</h3>", text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$',  r"<h2 style='color:#60a5fa; font-size:18px; font-weight:700; margin:16px 0 8px 0;'>\1</h2>", text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$',   r"<h1 style='color:#60a5fa; font-size:20px; font-weight:700; margin:20px 0 10px 0;'>\1</h1>", text, flags=re.MULTILINE)

        # Horizontal rules
        text = re.sub(r'^(?:---|\*\*\*|___)\s*$', r"<hr style='border:none; border-top:1px solid #334155; margin:12px 0;'>", text, flags=re.MULTILINE)

        # Blockquotes
        text = re.sub(
            r'^&gt; (.+)$',
            r"<blockquote style='border-left:3px solid #60a5fa; padding-left:12px; color:#cbd5e1; margin:8px 0; font-style:italic;'>\1</blockquote>",
            text, flags=re.MULTILINE
        )

        # Tables  |col|col|
        def render_table(m: re.Match) -> str:
            rows = [r.strip() for r in m.group(0).strip().splitlines() if r.strip()]
            html_rows = []
            for i, row in enumerate(rows):
                if re.match(r'^[\|\s\-:]+$', row):
                    continue
                cells = [c.strip() for c in row.strip('|').split('|')]
                tag = 'th' if i == 0 else 'td'
                style = "style='padding:6px 12px; border:1px solid #334155; color:#e2e8f0;'"
                html_rows.append('<tr>' + ''.join(f'<{tag} {style}>{c}</{tag}>' for c in cells) + '</tr>')
            return (
                "<table style='border-collapse:collapse; margin:8px 0; font-size:14px;'>"
                + ''.join(html_rows)
                + "</table>"
            )
        text = re.sub(r'((?:^\|.+\|\s*\n?)+)', render_table, text, flags=re.MULTILINE)

        # Numbered lists — group consecutive items into one <ol>
        def render_ol(m: re.Match) -> str:
            items = re.findall(r'^\d+\. (.+)$', m.group(0), flags=re.MULTILINE)
            lis = ''.join(f'<li style="margin-left:20px; margin-bottom:4px;">{item}</li>' for item in items)
            return f"<ol style='padding:0; margin:4px 0;'>{lis}</ol>"
        text = re.sub(r'((?:^\d+\. .+\n?)+)', render_ol, text, flags=re.MULTILINE)

        # Bullet lists — group consecutive items into one <ul>
        def render_ul(m: re.Match) -> str:
            items = re.findall(r'^\s*[-*] (.+)$', m.group(0), flags=re.MULTILINE)
            lis = ''.join(f'<li style="margin-left:20px; margin-bottom:4px;">{item}</li>' for item in items)
            return f"<ul style='list-style-type:disc; padding:0; margin:4px 0;'>{lis}</ul>"
        text = re.sub(r'((?:^\s*[-*] .+\n?)+)', render_ul, text, flags=re.MULTILINE)

        # Links
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" style="color:#60a5fa; text-decoration:underline;">\1</a>', text)

        # Restore escaped characters
        text = re.sub(r'\x01(\d+)\x01', lambda m: chr(int(m.group(1))), text)

        # Restore code blocks
        def restore_code(m: re.Match) -> str:
            return code_blocks[int(m.group(1))]
        text = re.sub(r'\x00CODE(\d+)\x00', restore_code, text)

        # Newlines to <br> (skip inside block elements)
        text = re.sub(r'\n', '<br>', text)

        return text

if _HAS_QT:
    class ResponseThread(QThread):
        result_ready = Signal(str, str)
        error_occurred = Signal(str)

        def __init__(self, assistant: "AtlasAI", user_text: str):
            super().__init__()
            self.assistant = assistant
            self.user_text = user_text

        def run(self) -> None:
            try:
                answer, details = self.assistant.respond_with_details(self.user_text)
                self.result_ready.emit(answer, details)
            except Exception as exc:
                self.error_occurred.emit(str(exc))

    class ChatBubble(QWidget):
        def __init__(self, assistant: "AtlasAI", role: str, text: str, details: str = ""):
            super().__init__()
            self.assistant = assistant
            self.role = role.lower()
            
            # Create a container layout that handles left/right alignment
            container_layout = QHBoxLayout(self)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            
            # Add stretch to left or right depending on role
            if self.role != "atlas":
                container_layout.addStretch()
            
            # Main message bubble
            bubble_widget = QWidget()
            bubble_layout = QVBoxLayout(bubble_widget)
            bubble_layout.setSpacing(8)
            bubble_layout.setContentsMargins(16, 14, 16, 14)
            
            # Role indicator
            role_label = QLabel(role)
            if self.role == "atlas":
                role_label.setStyleSheet("color: #60a5fa; font-weight: 700; font-size: 13px; margin-bottom: 2px;")
            else:
                role_label.setStyleSheet("color: #94a3b8; font-weight: 700; font-size: 13px; margin-bottom: 2px;")
            bubble_layout.addWidget(role_label)
            
            # Message content
            for widget in self._render_message(text):
                bubble_layout.addWidget(widget)
            
            # Details toggle
            if details:
                self.toggle_button = QPushButton("Show details")
                self.toggle_button.setStyleSheet(
                    "QPushButton { background: #334155; color: #cbd5e1; border-radius: 4px; padding: 4px 8px; font-size: 12px; }"
                    "QPushButton:hover { background: #475569; }"
                )
                self.toggle_button.setMaximumWidth(100)
                self.details_container = QWidget()
                details_layout = QVBoxLayout(self.details_container)
                details_layout.setContentsMargins(0, 0, 0, 0)
                for widget in self._render_message(details):
                    details_layout.addWidget(widget)
                self.details_container.setVisible(False)
                self.toggle_button.clicked.connect(self._toggle_details)
                bubble_layout.addWidget(self.toggle_button)
                bubble_layout.addWidget(self.details_container)
            
            # Bubble styling based on role
            if self.role == "atlas":
                bubble_widget.setStyleSheet(
                    "QWidget { border-radius: 12px; background: #1e293b; border: 1px solid #334155; margin: 8px 0px 8px 24px; }"
                )
            else:
                bubble_widget.setStyleSheet(
                    "QWidget { border-radius: 12px; background: #2563eb; border: none; margin: 8px 24px 8px 0px; }"
                )
            
            container_layout.addWidget(bubble_widget, 0, Qt.AlignTop)
            
            # Add stretch to right or left depending on role
            if self.role == "atlas":
                container_layout.addStretch() 

        def _render_message(self, text: str) -> list:
            import html as htmllib
            widgets = []
            # Split on code blocks
            parts = re.split(r'(```(?:\w+)?\n?.*?```)', text, flags=re.DOTALL)
            for part in parts:
                code_match = re.match(r'```(\w+)?\n?(.*?)```', part, flags=re.DOTALL)
                if code_match:
                    lang = code_match.group(1) or "plaintext"
                    code = code_match.group(2).rstrip()

                    # Wrapper widget for code block
                    wrapper = QWidget()
                    wrapper.setStyleSheet("QWidget { background: transparent; margin: 12px 0px; }")
                    wrapper_layout = QVBoxLayout(wrapper)
                    wrapper_layout.setContentsMargins(0, 0, 0, 0)
                    wrapper_layout.setSpacing(0)

                    # Top bar with language label and copy button
                    top_bar = QWidget()
                    top_bar.setStyleSheet("QWidget { background: #0d1117; border-radius: 6px 6px 0px 0px; border-bottom: 1px solid #30363d; }")
                    top_bar_layout = QHBoxLayout(top_bar)
                    top_bar_layout.setContentsMargins(12, 8, 12, 8)
                    
                    lang_label = QLabel(lang)
                    lang_label.setStyleSheet("color: #79c0ff; font-size: 11px; font-weight: 600; font-family: monospace;")
                    top_bar_layout.addWidget(lang_label)
                    top_bar_layout.addStretch()
                    
                    copy_btn = QPushButton("Copy")
                    copy_btn.setFixedSize(60, 28)
                    copy_btn.setStyleSheet(
                        "QPushButton { background: #238636; color: #f0f6fc; border-radius: 4px; font-size: 12px; padding: 2px; font-weight: 600; border: none; }"
                        "QPushButton:hover { background: #2da644; }"
                    )
                    copy_btn.clicked.connect(lambda checked, c=code: self._copy_code(c, copy_btn))
                    top_bar_layout.addWidget(copy_btn)
                    wrapper_layout.addWidget(top_bar)

                    # Code text area
                    code_edit = QTextEdit()
                    code_edit.setReadOnly(True)
                    code_edit.setPlainText(code)
                    code_edit.setStyleSheet(
                        "QTextEdit { background-color: #0d1117 !important; color: #c9d1d9 !important; "
                        "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 12px; "
                        "border-radius: 0px 0px 6px 6px; border: none; padding: 12px; line-height: 1.5; }"
                    )
                    code_edit.setMinimumHeight(80)
                    code_edit.setMaximumHeight(500)
                    code_edit.document().contentsChanged.connect(
                        lambda: code_edit.setFixedHeight(
                            min(int(code_edit.document().size().height()) + 24, 500)
                        )
                    )
                    wrapper_layout.addWidget(code_edit)
                    widgets.append(wrapper)
                else:
                    if not part.strip():
                        continue
                    rendered = self.assistant._render_markdown_for_gui(part) if hasattr(self.assistant, '_render_markdown_for_gui') else htmllib.escape(part).replace('\n', '<br>')
                    label = QLabel(f"<div style='font-size:15px; color:#e2e8f0; line-height:1.8; letter-spacing: 0.2px;'>{rendered}</div>")
                    label.setTextFormat(Qt.RichText)
                    label.setWordWrap(True)
                    label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
                    widgets.append(label)
            return widgets

        def _copy_code(self, code: str, button: "QPushButton") -> None:
            QApplication.clipboard().setText(code)
            original_text = button.text()
            button.setText("✓ Copied")
            button.setStyleSheet(
                "QPushButton { background: #238636; color: #f0f6fc; border-radius: 4px; font-size: 12px; padding: 2px; font-weight: 600; border: none; }"
            )
            QTimer.singleShot(2000, lambda: (button.setText(original_text), button.setStyleSheet(
                "QPushButton { background: #238636; color: #f0f6fc; border-radius: 4px; font-size: 12px; padding: 2px; font-weight: 600; border: none; }"
                "QPushButton:hover { background: #2da644; }"
            )))

        def _toggle_details(self) -> None:
            visible = not self.details_container.isVisible()
            self.details_container.setVisible(visible)
            self.toggle_button.setText("Hide details" if visible else "Show details")


    class DraggableMemoryListWidget(QListWidget):
        """A QListWidget whose items can be dragged out as [Context Note: …] text injects."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setDragEnabled(True)
            self.setSelectionMode(QListWidget.SingleSelection)
            self.setStyleSheet(
                "QListWidget { background: #1e293b; color: #e2e8f0; border: none; "
                "border-radius: 6px; font-size: 13px; }"
                "QListWidget::item { padding: 6px 10px; border-bottom: 1px solid #334155; }"
                "QListWidget::item:selected { background: #334155; color: #f8fafc; }"
                "QListWidget::item:hover { background: #263548; }"
            )

        def startDrag(self, supported_actions) -> None:  # type: ignore[override]
            item = self.currentItem()
            if item is None:
                return
            raw_text = item.data(Qt.UserRole) or item.text()
            inject_text = f"[Context Note: {raw_text}]"

            mime = QMimeData()
            mime.setText(inject_text)

            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.exec(Qt.CopyAction)

    class MemoryInsightWindow(QWidget):
        """Floating panel showing memory engrams; items can be dragged into the chat input."""

        def __init__(self, assistant: "AtlasAI", parent=None):
            super().__init__(parent, Qt.Window | Qt.Tool)
            self.assistant = assistant
            self.setWindowTitle("Memory Engrams")
            self.resize(420, 480)
            self.setMinimumSize(320, 260)
            self.setStyleSheet(
                "QWidget { background: #0f172a; color: #e2e8f0; }"
                "QPushButton { background: #3b82f6; color: #fff; border-radius: 6px; "
                "padding: 6px 12px; font-weight: 600; border: none; }"
                "QPushButton:hover { background: #2563eb; }"
            )

            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)

            # Header row
            header = QHBoxLayout()
            title = QLabel("🧠 Memory Engrams")
            title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f8fafc;")
            hint = QLabel("Drag any entry into the chat input")
            hint.setStyleSheet("font-size: 11px; color: #64748b;")
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setFixedHeight(28)
            refresh_btn.clicked.connect(self.populate)
            header.addWidget(title)
            header.addStretch()
            header.addWidget(refresh_btn)
            layout.addLayout(header)
            layout.addWidget(hint)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #334155;")
            layout.addWidget(sep)

            self.list_widget = DraggableMemoryListWidget()
            layout.addWidget(self.list_widget)

            self.status_label = QLabel("")
            self.status_label.setStyleSheet("font-size: 11px; color: #64748b;")
            layout.addWidget(self.status_label)

            self.populate()

        def populate(self) -> None:
            self.list_widget.clear()
            entries = self.assistant.memory.entries
            if not entries:
                item = QListWidgetItem("(memory is empty)")
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                self.list_widget.addItem(item)
                self.status_label.setText("0 engrams")
                return

            now = time.time()
            scored = []
            for entry in entries:
                age = now - entry.get("timestamp", now)
                decay = math.exp(-age / HALF_LIFE_SECONDS)
                type_weight = ENGRAM_TYPE_WEIGHTS.get(entry.get("tag", "fact"), 1.0)
                score = float(entry.get("weight", 1.0)) * type_weight * decay
                scored.append((score, entry))
            scored.sort(reverse=True, key=lambda x: x[0])

            for score, entry in scored:
                tag = entry.get("tag", "fact")
                text = entry.get("text", "").strip()
                if not text:
                    continue
                display = f"[{tag}] {text}"
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, text)  # raw text for the inject
                item.setToolTip(
                    f"Tag: {tag}  |  Score: {score:.3f}  |  "
                    f"Age: {int((now - entry.get('timestamp', now)) / 3600)}h\n"
                    "Drag into chat input to inject as [Context Note: …]"
                )
                self.list_widget.addItem(item)

            self.status_label.setText(f"{len(scored)} engrams — sorted by relevance score")

def render_markdown_to_html(text: str) -> str:
    """
    Robust markdown → HTML converter for the Atlas chat log.
    Key fix: block-level elements are stashed before newline conversion
    so <br> never gets injected inside <pre>, <ul>, <ol>, <table> etc.
    """
    import html as _html

    # ── 0. Stash fenced code blocks ──────────────────────────────────────
    _blocks: list = []

    def _stash_code(m: re.Match) -> str:
        lang = (m.group(1) or "").strip() or "plaintext"
        code = _html.escape(m.group(2).rstrip())
        block = (
            f"<div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; "
            f"margin:10px 0; overflow:auto;'>"
            f"<div style='padding:6px 14px; background:#161b22; border-bottom:1px solid #30363d; "
            f"color:#79c0ff; font-size:11px; font-family:monospace; font-weight:600;'>{_html.escape(lang)}</div>"
            f"<pre style='margin:0; padding:14px; color:#c9d1d9; font-family:monospace; "
            f"font-size:13px; white-space:pre-wrap; line-height:1.55;'>{code}</pre>"
            f"</div>"
        )
        _blocks.append(block)
        return f"\x00BLK{len(_blocks)-1}\x00"

    text = re.sub(r"```(\w*)\n?([\s\S]*?)```", _stash_code, text)

    # ── 1. Escape remaining HTML ──────────────────────────────────────────
    text = _html.escape(text)

    # ── 2. Escaped markdown chars  e.g. \*  \_ ───────────────────────────
    text = re.sub(r"\\([*_~`#\-\[\]\\])", lambda m: f"\x01{ord(m.group(1))}\x01", text)

    # ── 3. Inline code ────────────────────────────────────────────────────
    text = re.sub(
        r"`([^`\n]+)`",
        r"<code style='background:#1e293b;color:#60a5fa;padding:2px 6px;"
        r"border-radius:3px;font-family:monospace;font-size:13px;'>\1</code>",
        text,
    )

    # ── 4. Bold+italic / bold / italic / strikethrough ────────────────────
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<b style='color:#f0f6fc;'>\1</b>", text)
    text = re.sub(r"__(.+?)__",         r"<b style='color:#f0f6fc;'>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",         r"<i style='color:#cbd5e1;'>\1</i>", text)
    text = re.sub(r"_(.+?)_",           r"<i style='color:#cbd5e1;'>\1</i>", text)
    text = re.sub(r"~~(.+?)~~",         r"<s style='color:#64748b;'>\1</s>", text)

    # ── 5. Headers ────────────────────────────────────────────────────────
    text = re.sub(r"^### (.+)$", r"<h3 style='color:#818cf8;font-size:15px;font-weight:700;"
                  r"margin:12px 0 4px;'>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$",  r"<h2 style='color:#818cf8;font-size:17px;font-weight:700;"
                  r"margin:14px 0 6px;'>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",   r"<h1 style='color:#818cf8;font-size:20px;font-weight:700;"
                  r"margin:16px 0 8px;'>\1</h1>", text, flags=re.MULTILINE)

    # ── 6. Horizontal rules ───────────────────────────────────────────────
    text = re.sub(r"^(?:---|\*\*\*|___)\s*$",
                  r"<hr style='border:none;border-top:1px solid #334155;margin:12px 0;'>",
                  text, flags=re.MULTILINE)

    # ── 7. Blockquotes ────────────────────────────────────────────────────
    text = re.sub(
        r"^&gt; (.+)$",
        r"<blockquote style='border-left:3px solid #6366f1;padding-left:12px;"
        r"color:#94a3b8;margin:8px 0;font-style:italic;'>\1</blockquote>",
        text, flags=re.MULTILINE,
    )

    # ── 8. Tables ─────────────────────────────────────────────────────────
    def _render_table(m: re.Match) -> str:
        rows = [r.strip() for r in m.group(0).strip().splitlines() if r.strip()]
        html_rows = []
        for i, row in enumerate(rows):
            if re.match(r"^[\|\s\-:]+$", row):
                continue
            cells = [c.strip() for c in row.strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            style = "style='padding:6px 12px;border:1px solid #334155;color:#e2e8f0;'"
            html_rows.append("<tr>" + "".join(f"<{tag} {style}>{c}</{tag}>" for c in cells) + "</tr>")
        return (
            "<table style='border-collapse:collapse;margin:8px 0;font-size:14px;'>"
            + "".join(html_rows) + "</table>\n"
        )
    text = re.sub(r"((?:^\|.+\|\s*\n?)+)", _render_table, text, flags=re.MULTILINE)

    # ── 9. Numbered lists ─────────────────────────────────────────────────
    def _render_ol(m: re.Match) -> str:
        items = re.findall(r"^\d+\. (.+)$", m.group(0), flags=re.MULTILINE)
        lis = "".join(f"<li style='margin-bottom:4px;'>{item}</li>" for item in items)
        return f"<ol style='padding-left:24px;margin:6px 0;'>{lis}</ol>\n"
    text = re.sub(r"((?:^\d+\. .+\n?)+)", _render_ol, text, flags=re.MULTILINE)

    # ── 10. Bullet lists ──────────────────────────────────────────────────
    def _render_ul(m: re.Match) -> str:
        items = re.findall(r"^\s*[-*] (.+)$", m.group(0), flags=re.MULTILINE)
        lis = "".join(f"<li style='margin-bottom:4px;'>{item}</li>" for item in items)
        return f"<ul style='list-style-type:disc;padding-left:24px;margin:6px 0;'>{lis}</ul>\n"
    text = re.sub(r"((?:^\s*[-*] .+\n?)+)", _render_ul, text, flags=re.MULTILINE)

    # ── 11. Links ─────────────────────────────────────────────────────────
    text = re.sub(
        r"\[(.+?)\]\((.+?)\)",
        r'<a href="\2" style="color:#60a5fa;text-decoration:underline;">\1</a>',
        text,
    )

    # ── 12. Restore escaped chars ─────────────────────────────────────────
    text = re.sub(r"\x01(\d+)\x01", lambda m: _html.escape(chr(int(m.group(1)))), text)

    # ── 13. Newlines → <br>  (only plain text lines; block tags already end with \n) ──
    # Lines that are already wrapped in block-level tags don't need <br>
    _BLOCK_TAGS = re.compile(r"^\s*<(?:h[1-6]|hr|ul|ol|li|table|tr|th|td|blockquote|div|pre)", re.I)
    result_lines = []
    for line in text.split("\n"):
        if _BLOCK_TAGS.match(line) or line.strip() == "":
            result_lines.append(line)
        else:
            result_lines.append(line + "<br>")
    text = "\n".join(result_lines)

    # ── 14. Restore stashed code blocks ──────────────────────────────────
    def _restore(m: re.Match) -> str:
        return _blocks[int(m.group(1))]
    text = re.sub(r"\x00BLK(\d+)\x00", _restore, text)

    return text


class MarkdownChatBubble(QWidget):
    def __init__(self, text: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.is_user = is_user

        # Horizontal layout alignment wrapper
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)

        # Main bubble frame
        self.bubble = QFrame()
        
        # Style layout profiles based on role 
        if is_user:
            # User profile (Aligned right, dark slate background)
            self.bubble.setStyleSheet("""
                QFrame {
                    background-color: #2D3748;
                    border-radius: 14px;
                    border-bottom-right-radius: 2px;
                }
            """)
            text_color = "#FFFFFF"
            layout.addStretch(1)
            layout.addWidget(self.bubble)
        else:
            # Assistant profile (Aligned left, clean ash background)
            self.bubble.setStyleSheet("""
                QFrame {
                    background-color: #EDF2F7;
                    border-radius: 14px;
                    border-bottom-left-radius: 2px;
                }
            """)
            text_color = "#1A202C"
            layout.addWidget(self.bubble)
            layout.addStretch(1)

        # Bubble internal padding layout
        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)

        # QTextBrowser acts as our Markdown render viewport
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setUndoRedoEnabled(False)
        self.browser.setReadOnly(True)
        
        # Transparent canvas rules out parent inheritance styling glitches
        self.browser.setStyleSheet(f"""
            QTextBrowser {{
                background: transparent;
                border: none;
                color: {text_color};
            }}
        """)

        # Process text and update content window
        html_content = render_markdown_to_html(text)
        self.browser.setHtml(html_content)

        # Dynamic multi-line height computation framework
        doc = self.browser.document()
        doc.setTextWidth(450) # Bind text tracking constraint boundary layout 
        ideal_width = int(doc.idealWidth()) + 25
        ideal_height = int(doc.size().height()) + 10

        # Enforce elastic sizing guidelines
        self.browser.setFixedSize(QSize(min(ideal_width, 450), ideal_height))
        self.bubble.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        bubble_layout.addWidget(self.browser)

class AtlasGUI(QWidget):
    def __init__(self, assistant: AtlasAI):
        super().__init__()
        self.assistant = assistant
        self.setWindowTitle("Atlas")
        self.setGeometry(80, 60, 1100, 700)
        self.setMinimumSize(800, 520)
     
        # ── Global stylesheet ─────────────────────────────────────────────
        BG_DEEP   = "#09090b"   # near-black main bg
        BG_SIDE   = "#111113"   # sidebar bg — slightly lifted
        BG_CARD   = "#18181b"   # card / bubble bg
        BG_INPUT  = "#1c1c1f"   # input field bg
        BORDER    = "#27272a"   # subtle border colour
        ACCENT    = "#6366f1"   # indigo accent
        ACCENT_HV = "#818cf8"   # hover accent
        TEXT_PRI  = "#fafafa"
        TEXT_SEC  = "#a1a1aa"
        TEXT_MUT  = "#52525b"

        self.setStyleSheet(f"""
            QWidget {{ background: {BG_DEEP}; color: {TEXT_PRI}; font-family: 'JetBrains Mono', 'Fira Code', monospace; }}
            QPushButton {{ background: {BG_CARD}; color: {TEXT_PRI}; border-radius: 6px;
                           padding: 7px 14px; font-weight: 600; border: 1px solid {BORDER}; font-size: 13px; }}
            QPushButton:hover {{ background: {BORDER}; border-color: {ACCENT}; color: {ACCENT_HV}; }}
            QPushButton:pressed {{ background: {ACCENT}; color: #fff; border-color: {ACCENT}; }}
            QPushButton#accentBtn {{ background: {ACCENT}; color: #fff; border: none; }}
            QPushButton#accentBtn:hover {{ background: {ACCENT_HV}; }}
            QPushButton#sidebarBtn {{ background: transparent; color: {TEXT_SEC}; border: none;
                                      border-radius: 6px; padding: 8px 12px; text-align: left; font-size: 13px; }}
            QPushButton#sidebarBtn:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}
            QPushButton#sidebarBtn:pressed {{ background: {BORDER}; }}
            QLineEdit {{ background: {BG_INPUT}; color: {TEXT_PRI}; border: 1px solid {BORDER};
                         border-radius: 10px; padding: 10px 16px; font-size: 14px; }}
            QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background: {TEXT_MUT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QListWidget {{ background: transparent; border: none; color: {TEXT_SEC}; font-size: 13px; }}
            QListWidget::item {{ padding: 8px 10px; border-radius: 6px; margin: 1px 0; }}
            QListWidget::item:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}
            QListWidget::item:selected {{ background: {BG_CARD}; color: {TEXT_PRI}; border: none; }}
            QMenu {{ background: {BG_CARD}; color: {TEXT_PRI}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
            QMenu::item {{ padding: 6px 20px; border-radius: 4px; }}
            QMenu::item:selected {{ background: {BORDER}; }}
            QDialog {{ background: {BG_DEEP}; }}
            QTextEdit {{ background: {BG_CARD}; color: {TEXT_SEC}; border: 1px solid {BORDER}; border-radius: 6px; }}
            QComboBox {{ background: {BG_CARD}; color: {TEXT_PRI}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 10px; }}
        """)

        # ── Root layout: sidebar | chat ───────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ═══════════════════════════════════════════════════════
        #  LEFT SIDEBAR
        # ═══════════════════════════════════════════════════════
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet(f"QWidget {{ background: {BG_SIDE}; border-right: 1px solid {BORDER}; }}")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 14, 10, 14)
        sidebar_layout.setSpacing(4)

        # Brand
        brand = QLabel("Atlas")
        brand.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {TEXT_PRI}; letter-spacing: -0.5px; padding: 4px 6px 10px 6px;")
        sidebar_layout.addWidget(brand)

        # New chat button
        new_chat_btn = QPushButton("  + New chat")
        new_chat_btn.setObjectName("sidebarBtn")
        new_chat_btn.setStyleSheet(
            f"QPushButton {{ background: {BG_CARD}; color: {TEXT_PRI}; border: 1px solid {BORDER}; "
            f"border-radius: 8px; padding: 9px 14px; font-size: 13px; font-weight: 600; text-align: left; }}"
            f"QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT_HV}; }}"
        )
        new_chat_btn.clicked.connect(self._on_new_chat)
        sidebar_layout.addWidget(new_chat_btn)
        sidebar_layout.addSpacing(10)

        # Chats section label
        chats_label = QLabel("Recents")
        chats_label.setStyleSheet(f"color: {TEXT_MUT}; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; padding: 0 6px;")
        sidebar_layout.addWidget(chats_label)

        # Chat history list
        self.chat_history_list = QListWidget()
        self.chat_history_list.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; }}"
            f"QListWidget::item {{ padding: 7px 8px; border-radius: 6px; color: {TEXT_SEC}; font-size: 13px; }}"
            f"QListWidget::item:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}"
            f"QListWidget::item:selected {{ background: {BG_CARD}; color: {TEXT_PRI}; }}"
        )
        self.chat_history_list.itemClicked.connect(self._on_chat_history_item_clicked)
        sidebar_layout.addWidget(self.chat_history_list, 1)  # stretch

        self._refresh_chat_list()

        # ── Sidebar footer ────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-height: 1px;")
        sidebar_layout.addWidget(sep)
        sidebar_layout.addSpacing(6)

        # Memory engram toggle button
        self.memory_btn = QPushButton("🧠  Memory Engrams")
        self.memory_btn.setObjectName("sidebarBtn")
        self.memory_btn.setCheckable(True)
        self.memory_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_SEC}; border: none; "
            f"border-radius: 6px; padding: 8px 10px; font-size: 13px; text-align: left; }}"
            f"QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}"
            f"QPushButton:checked {{ background: {BG_CARD}; color: {ACCENT_HV}; }}"
        )
        self.memory_btn.toggled.connect(self._toggle_memory_panel)
        self._memory_insight_window: Optional["MemoryInsightWindow"] = None
        sidebar_layout.addWidget(self.memory_btn)
        sidebar_layout.addSpacing(4)

        # Model section
        model_sep = QFrame()
        model_sep.setFrameShape(QFrame.HLine)
        model_sep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-height: 1px;")
        sidebar_layout.addWidget(model_sep)
        sidebar_layout.addSpacing(6)

        # Workspace button — above model info
        ws_root = self.assistant.workspace.root
        ws_label_text = f"📁  {os.path.basename(ws_root) or ws_root}" if ws_root else "📁  Set Workspace"
        self.workspace_btn = QPushButton(ws_label_text)
        self.workspace_btn.setObjectName("sidebarBtn")
        self.workspace_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_SEC}; border: none; "
            f"border-radius: 6px; padding: 8px 10px; font-size: 13px; text-align: left; }}"
            f"QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}"
            f"QPushButton:checked {{ background: {BG_CARD}; color: {ACCENT_HV}; }}"
        )
        self.workspace_btn.clicked.connect(self._on_set_workspace)
        sidebar_layout.addWidget(self.workspace_btn)
        sidebar_layout.addSpacing(4)

        ws_sep = QFrame()
        ws_sep.setFrameShape(QFrame.HLine)
        ws_sep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-height: 1px;")
        sidebar_layout.addWidget(ws_sep)
        sidebar_layout.addSpacing(6)

        model_name = os.path.basename(self.assistant.model_path) if self.assistant.model_path else "No model loaded"
        self.model_label = QLabel(f"⚙  {model_name[:28]}")
        self.model_label.setStyleSheet(f"color: {TEXT_MUT}; font-size: 11px; padding: 0 6px 2px 6px;")
        self.model_label.setWordWrap(True)
        sidebar_layout.addWidget(self.model_label)

        model_btn_row = QHBoxLayout()
        model_btn_row.setSpacing(6)
        load_model_btn = QPushButton("Load model")
        load_model_btn.setObjectName("sidebarBtn")
        load_model_btn.clicked.connect(self._on_load_model)
        unload_model_btn = QPushButton("Unload")
        unload_model_btn.setObjectName("sidebarBtn")
        unload_model_btn.clicked.connect(self._on_unload_model)
        model_btn_row.addWidget(load_model_btn)
        model_btn_row.addWidget(unload_model_btn)
        sidebar_layout.addLayout(model_btn_row)

        root.addWidget(sidebar)

        # ═══════════════════════════════════════════════════════
        #  MAIN AREA  (chat + input)
        # ═══════════════════════════════════════════════════════
        main_area = QWidget()
        main_area.setStyleSheet(f"QWidget {{ background: {BG_DEEP}; }}")
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Thin top bar with debug toggle
        topbar = QWidget()
        topbar.setFixedHeight(40)
        topbar.setStyleSheet(f"QWidget {{ background: {BG_DEEP}; border-bottom: 1px solid {BORDER}; }}")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(20, 0, 16, 0)
        topbar_row_label = QLabel("Atlas AI")
        topbar_row_label.setStyleSheet(f"color: {TEXT_MUT}; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;")
        debug_btn = QPushButton("Debug")
        debug_btn.setFixedSize(60, 26)
        debug_btn.setCheckable(True)
        debug_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_MUT}; border: 1px solid {BORDER}; "
            f"border-radius: 5px; font-size: 11px; padding: 0; }}"
            f"QPushButton:hover {{ color: {TEXT_PRI}; border-color: {TEXT_MUT}; }}"
            f"QPushButton:checked {{ color: {ACCENT_HV}; border-color: {ACCENT}; }}"
        )
        debug_btn.toggled.connect(self._toggle_debug_panel)
        topbar_layout.addWidget(topbar_row_label)
        topbar_layout.addStretch()
        topbar_layout.addWidget(debug_btn)
        main_layout.addWidget(topbar)

        # Debug dialog
        self.debug_dialog = QDialog(self)
        self.debug_dialog.setWindowTitle("Atlas Debug")
        self.debug_dialog.resize(700, 500)
        debug_layout = QVBoxLayout(self.debug_dialog)
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        debug_layout.addWidget(self.debug_text)
        self.debug_dialog.setLayout(debug_layout)
        self._current_stream_bubble = None

        # Chat area with scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }") # Clean borderless look
    
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(5, 5, 5, 5)
        self.chat_layout.setSpacing(10) # Keeps spacing tight between message clusters
        self.chat_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area, 1)

        # Input area
        input_wrapper = QWidget()
        input_wrapper.setStyleSheet(f"QWidget {{ background: {BG_DEEP}; }}")
        input_outer = QVBoxLayout(input_wrapper)
        input_outer.setContentsMargins(60, 10, 60, 20)
        input_outer.setSpacing(6)

        input_box = QWidget()
        input_box.setStyleSheet(
            f"QWidget {{ background: {BG_INPUT}; border: 1px solid {BORDER}; "
            f"border-radius: 12px; }}"
        )
        input_box_layout = QHBoxLayout(input_box)
        input_box_layout.setContentsMargins(14, 6, 8, 6)
        input_box_layout.setSpacing(8)

        # DragDropLineEdit — accepts [Context Note: …] drops from MemoryInsightWindow
        ACCENT_LOCAL = ACCENT
        ACCENT_HV_LOCAL = ACCENT_HV
        BORDER_LOCAL = BORDER
        BG_INPUT_LOCAL = BG_INPUT
        TEXT_PRI_LOCAL = TEXT_PRI

        class DragDropLineEdit(QLineEdit):
            def customDragEnterEvent(self, event) -> None:  # type: ignore[override]
                if event.mimeData().hasText() and event.mimeData().text().startswith("[Context Note:"):
                    event.acceptProposedAction()
                    self.setStyleSheet(
                        f"QLineEdit {{ background: transparent; color: {TEXT_PRI_LOCAL}; "
                        f"border: none; padding: 6px 0; font-size: 14px; }}"
                    )
                else:
                    event.ignore()

            def dragEnterEvent(self, event) -> None:  # type: ignore[override]
                self.customDragEnterEvent(event)

            def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
                self.setStyleSheet(
                    f"QLineEdit {{ background: transparent; color: {TEXT_PRI_LOCAL}; "
                    f"border: none; padding: 6px 0; font-size: 14px; }}"
                )

            def customDropEvent(self, event) -> None:  # type: ignore[override]
                if event.mimeData().hasText():
                    inject = event.mimeData().text()
                    current = self.text()
                    sep = " " if current and not current.startswith(" ") else ""
                    self.setText(inject + sep + current)
                    self.setCursorPosition(len(self.text()))
                    event.acceptProposedAction()
                self.dragLeaveEvent(event)

            def dropEvent(self, event) -> None:  # type: ignore[override]
                self.customDropEvent(event)

        self.input_line = DragDropLineEdit()
        self.input_line.setAcceptDrops(True)
        self.input_line.setPlaceholderText("Message Atlas…")
        self.input_line.setStyleSheet(
            f"QLineEdit {{ background: transparent; color: {TEXT_PRI}; border: none; "
            f"padding: 6px 0; font-size: 14px; }}"
        )
        self.input_line.returnPressed.connect(self.on_send)

        self.send_button = QPushButton("↑")
        self.send_button.setObjectName("accentBtn")
        self.send_button.setFixedSize(36, 36)
        self.send_button.setStyleSheet(
            f"QPushButton {{ background: {ACCENT}; color: #fff; border-radius: 8px; "
            f"font-size: 18px; font-weight: 700; border: none; padding: 0; }}"
            f"QPushButton:hover {{ background: {ACCENT_HV}; }}"
            f"QPushButton:disabled {{ background: {BORDER}; color: {TEXT_MUT}; }}"
        )
        self.send_button.clicked.connect(self.on_send)

        input_box_layout.addWidget(self.input_line, 1)
        input_box_layout.addWidget(self.send_button)
        input_outer.addWidget(input_box)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_MUT}; font-size: 11px; padding: 0 4px;")
        input_outer.addWidget(self.status_label)

        main_layout.addWidget(input_wrapper)
        root.addWidget(main_area, 1)

    def _refresh_chat_list(self) -> None:
        """Populate the sidebar chat history list from disk."""
        self.chat_history_list.clear()
        if not os.path.isdir(CHAT_LOG_DIR):
            return
        files = sorted(
            [f for f in os.listdir(CHAT_LOG_DIR) if f.endswith(".jsonl")],
            reverse=True,
        )
        for fname in files[:40]:
            display = os.path.splitext(fname)[0].replace("_", " ")
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, os.path.join(CHAT_LOG_DIR, fname))
            self.chat_history_list.addItem(item)

    def _on_chat_history_item_clicked(self, item: "QListWidgetItem") -> None:
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            response = self.assistant.load_chat_history_file(path)
            self._clear_chat_view()
            # Re-render loaded history
            for entry in self.assistant.history:
                role = "You" if entry.get("role") == "user" else "Atlas"
                self._append_chat(role, entry.get("message", ""))
            self._append_chat("Atlas", response)

    def _append_chat(self, role: str, text: str, details: str = "") -> None:
        bubble = ChatBubble(self.assistant, role, text, details)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def _append_debug(self, text: str) -> None:
        if hasattr(self, "debug_text") and self.debug_text is not None:
            self.debug_text.append(text)

    def _toggle_debug_panel(self, checked: bool) -> None:
        if checked:
            self.debug_dialog.show()
        else:
            self.debug_dialog.hide()

    def _toggle_memory_panel(self, checked: bool) -> None:
        if checked:
            if self._memory_insight_window is None:
                self._memory_insight_window = MemoryInsightWindow(self.assistant, parent=None)
                self._memory_insight_window.destroyed.connect(
                    lambda: setattr(self, "_memory_insight_window", None)
                )
            self._memory_insight_window.populate()
            self._memory_insight_window.show()
            self._memory_insight_window.raise_()
        else:
            if self._memory_insight_window is not None:
                self._memory_insight_window.hide()

    def add_message(self, sender: str, text: str) -> None:
        if self.thinking_label:
            self.chat_layout.removeWidget(self.thinking_label)
            self.thinking_label.deleteLater()
            self.thinking_label = None

        is_user = (sender.lower() == "user" or sender.lower() == "you")
    
        # Initialize a permanent, non-streaming layout component
        bubble = StyledBubble(text, is_user=is_user)
    
        # Safely insert right above your trailing alignment stretch spacer
        count = self.chat_layout.count()
        self.chat_layout.insertWidget(count - 1, bubble)
        self.scroll_to_bottom()

    def update_chat(self, sender: str, text: str) -> None:
        if self.thinking_label:
            self.chat_layout.removeWidget(self.thinking_label)
            self.thinking_label.deleteLater()
            self.thinking_label = None

        is_user = (sender.lower() == "user" or sender.lower() == "you")

        # Track down if our active stream bubble already exists in the layout view.
        # If it doesn't, create a new one instantly on the fly.
        if not hasattr(self, "_current_stream_bubble") or self._current_stream_bubble is None:
            self._current_stream_bubble = StyledBubble(text, is_user=is_user)
            count = self.chat_layout.count()
            self.chat_layout.insertWidget(count - 1, self._current_stream_bubble)
        else:
            # If it already exists, feed the updated complete text string into the HTML browser layout
            self._current_stream_bubble.update_text(text)

        self.scroll_to_bottom()


    def _on_set_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Workspace Directory",
            self.assistant.workspace.root or str(pathlib.Path.home()),
        )
        if not path:
            return
        result = self.assistant.workspace.set_root(path)
        self._append_chat("Atlas", result)
        self.assistant.history.append({"role": "assistant", "message": result})
        # Update sidebar button label
        label = f"📁  {os.path.basename(path) or path}"
        self.workspace_btn.setText(label)

    def _on_load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model",
            MODEL_SEARCH_DIR,
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if not path:
            return
        response = self.assistant.load_model(path)
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self.assistant.save_chat_history()
        self.model_label.setText(f"⚙  {os.path.basename(path)[:28]}")

    def _on_unload_model(self) -> None:
        self.assistant.llm = None
        self.assistant.model_path = None
        import gc; gc.collect()
        self.assistant.gpu_layers = 0
        self.model_label.setText("⚙  No model loaded")
        response = "No model loaded. Atlas is now in no-model mode."
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self.assistant.save_chat_history()

        if not hasattr(self, "_current_stream_bubble") or self._current_stream_bubble is None:
            self._current_stream_bubble = StyledBubble(text, is_user=is_user)
            count = self.chat_layout.count()
            self.chat_layout.insertWidget(count - 1, self._current_stream_bubble)
        else:
            # If it already exists, just pipe the updated live text chunk over
            self._current_stream_bubble.update_text(text)

        self.scroll_to_bottom()


    def _on_set_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Workspace Directory",
            self.assistant.workspace.root or str(pathlib.Path.home()),
        )
        if not path:
            return
        result = self.assistant.workspace.set_root(path)
        self._append_chat("Atlas", result)
        self.assistant.history.append({"role": "assistant", "message": result})
        # Update sidebar button label
        label = f"📁  {os.path.basename(path) or path}"
        self.workspace_btn.setText(label)

    def _on_load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model",
            MODEL_SEARCH_DIR,
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if not path:
            return
        response = self.assistant.load_model(path)
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self.assistant.save_chat_history()
        self.model_label.setText(f"⚙  {os.path.basename(path)[:28]}")

    def _on_unload_model(self) -> None:
        self.assistant.llm = None
        self.assistant.model_path = None
        import gc; gc.collect()
        self.assistant.gpu_layers = 0
        self.model_label.setText("⚙  No model loaded")
        response = "No model loaded. Atlas is now in no-model mode."
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self.assistant.save_chat_history()

    def _on_select_model(self, model_path: str) -> None:
        response = self.assistant.load_model(model_path)
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self.assistant.save_chat_history()
        self.model_label.setText(f"⚙  {os.path.basename(model_path)[:28]}")

    def _on_save_chat_as(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Chat As", "Enter chat name (1-3 words):")
        if not ok or not name.strip():
            return
        response = self.assistant.save_chat_history(name.strip())
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self._refresh_chat_list()

    def _on_save_chat(self) -> None:
        response = self.assistant.save_chat_history()
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})
        self._refresh_chat_list()

    def _clear_chat_view(self) -> None:
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.chat_layout.addStretch()

    def _on_new_chat(self) -> None:
        if self.assistant.history:
            try:
                self.assistant.save_chat_history()
            except Exception:
                pass
        self.assistant.history = []
        self.assistant.chat_filename = None
        self._clear_chat_view()
        self._append_chat("Atlas", "Started a new chat.")
        self.status_label.setText("New chat ready")
        self._refresh_chat_list()

    def _on_load_chat(self) -> None:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Saved Chat",
            CHAT_LOG_DIR,
            "Chat Files (*.jsonl);;All Files (*)",
        )
        if not path:
            return
        response = self.assistant.load_chat_history_file(path)
        self._append_chat("Atlas", response)
        self.assistant.history.append({"role": "assistant", "message": response})

    def on_send(self) -> None:
        user_text = self.input_line.text().strip()
        if not user_text:
            return

        self._append_chat("You", user_text)
        self.current_user_text = user_text
        self.assistant.history.append({"role": "user", "message": user_text})
        self.input_line.clear()
        self.status_label.setText("Thinking…")
        self.send_button.setEnabled(False)
        QApplication.processEvents()

        command_response = self.assistant.handle_command(user_text)
        if command_response is not None:
            self._append_chat("Atlas", command_response)
            self.assistant.history.append({"role": "assistant", "message": command_response})
            self.assistant.save_chat_history()
            self.send_button.setEnabled(True)
            self.status_label.setText("")
            self._refresh_chat_list()
            return

        self._append_debug(f"[DEBUG] User input: {user_text}\n")
        self.worker = ResponseThread(self.assistant, user_text)
        self.worker.result_ready.connect(self._on_response_ready)
        self.worker.error_occurred.connect(self._on_response_error)
        self.worker.start()

    def _on_response_ready(self, answer: str, details: str) -> None:
        self.assistant.history.append({"role": "assistant", "message": answer})
        self.assistant.save_chat_history()
        self._append_chat("Atlas", answer, details)
        if self.assistant.last_prompt:
            self._append_debug(f"[DEBUG] Prompt sent:\n{self.assistant.last_prompt}\n")
        if self.assistant.last_raw_response:
            self._append_debug(f"[DEBUG] Raw model response:\n{self.assistant.last_raw_response}\n")
        self.send_button.setEnabled(True)
        self.status_label.setText("")
        self._refresh_chat_list()

    def _on_response_error(self, error_message: str) -> None:
        self._append_chat("Atlas", f"Error: {error_message}")
        self.send_button.setEnabled(True)
        self.status_label.setText("")



def select_model(models: List[str], fallback: Optional[str] = None) -> str:
    if not models and not fallback:
        raise FileNotFoundError("No GGUF models found.")
    if fallback:
        return fallback
    if len(models) == 1:
        return models[0]
    print("Available models:")
    for idx, path in enumerate(models, start=1):
        print(f"  {idx}. {os.path.basename(path)}")
    while True:
        choice = input("Choose model number (or press Enter for first): ").strip()
        if not choice:
            return models[0]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid selection.")


def select_model_gui(models: List[str], parent: Optional[QWidget] = None) -> Optional[str]:
    if not _HAS_QT:
        return None

    dialog = QDialog(parent)
    dialog.setWindowTitle("Select Model")
    layout = QVBoxLayout(dialog)
    label = QLabel("Choose a GGUF model or select No model:", dialog)
    layout.addWidget(label)

    combo = QComboBox(dialog)
    combo.addItem("0. No model")
    for idx, path in enumerate(models, start=1):
        combo.addItem(f"{idx}. {os.path.basename(path)}")
    layout.addWidget(combo)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    index = combo.currentIndex()
    if index == 0:
        return ""
    model_index = index - 1
    if 0 <= model_index < len(models):
        return models[model_index]
    return None


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python3 Atlas.py [--model PATH] [--cli]")
        print("       python3 AtlasAI.py [--model PATH] [--cli]")
        return

    model_path = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]

    if model_path and not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        sys.exit(1)

    if _HAS_QT and "--cli" not in sys.argv:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        if model_path is None:
            try:
                models = find_gguf_models()
            except Exception as exc:
                print(f"Error finding models: {exc}")
                sys.exit(1)
            model_path = select_model_gui(models)
            if model_path is None:
                print("Model selection cancelled.")
                sys.exit(0)

        assistant = AtlasAI(model_path=model_path, memory_path=MEMORY_FILE)
        window = AtlasGUI(assistant)
        window.show()
        window.raise_()
        window.activateWindow()
        QTimer.singleShot(0, window.activateWindow)
        sys.exit(app.exec())
    else:
        if model_path is None:
            models = find_gguf_models()
            model_path = select_model(models)
        assistant = AtlasAI(model_path=model_path, memory_path=MEMORY_FILE)
        if not _HAS_QT:
            print("PySide6 is not available, falling back to CLI mode.")
        assistant.run()

if __name__ == "__main__":
    main()