import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    
    # ── Environment Variables (only essentials) ───────────────────────────────
    gemini_api_key: str
    db_dir: str = "chroma_infohub_bgem3_full"
    corpus_path: str = "bgem3_corpus_full.pkl"
    
    # ── Model Configuration (static) ──────────────────────────────────────────
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    collection_name: str = "infohub_ka"
    
    # ── Retrieval Parameters (static) ─────────────────────────────────────────
    dense_top_k: int = 60
    rerank_top_n: int = 5
    fusion_weights: tuple = (0.3, 0.2, 0.5)  # dense, sparse, colbert
    
    # ── LLM Configuration (static) ────────────────────────────────────────────
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("GEMINI_API_KEY", "")
        db_dir = os.getenv("DB_DIR", "chroma_infohub_bgem3_full")
        corpus_path = os.getenv("CORPUS_PATH", "bgem3_corpus_full.pkl")
        
        return cls(
            gemini_api_key=api_key,
            db_dir=db_dir,
            corpus_path=corpus_path
        )
    
    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in environment variables.\n"
                "Please create a .env file with your API key."
            )
        
        if not os.path.exists(self.db_dir):
            raise FileNotFoundError(
                f"Vector database not found at '{self.db_dir}'.\n"
                f"Please run newvecdbcrawl.py first to ingest data."
            )
        
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(
                f"Corpus file not found at '{self.corpus_path}'.\n"
                f"Please run newvecdbcrawl.py first to ingest data."
            )
