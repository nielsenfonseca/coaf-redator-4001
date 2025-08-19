from pathlib import Path
from dotenv import load_dotenv
import os

# SUBA 3 níveis a partir de utils/ → coaf_redator → src → (raiz do projeto)
# antes: parents[2] (termina em .../src)  ➜  agora: parents[3] (raiz)
ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_DIR    = os.getenv("CHROMA_DB_DIR", str(ROOT / "chroma_db"))
HYBRID_ALPHA     = float(os.getenv("HYBRID_ALPHA", "0.5"))
TOP_K            = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "60"))

# Este agora aponta para docs/ na RAIZ do projeto (fora de src/)
NORM_4001_PDF    = str(ROOT / "docs/normativos/Carta_Circular_4001_BCB.pdf")
