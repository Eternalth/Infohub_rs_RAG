import pickle
from typing import Optional, List, Tuple

import chromadb
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from config import Config


class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[BGEM3FlagModel] = None
        self.reranker: Optional[FlagReranker] = None
        self.collection = None
        self.corpus: dict = {}
        
    def initialize(self) -> None:
        print("Loading BGE-M3 embedding model …")
        self.model = BGEM3FlagModel(self.config.embed_model, use_fp16=True)
        print("BGE-M3 loaded successfully!")
        
        print("Loading reranker model …")
        self.reranker = FlagReranker(self.config.rerank_model, use_fp16=True)
        print("Reranker loaded successfully!")
        
        print("Connecting to ChromaDB …")
        client = chromadb.PersistentClient(path=self.config.db_dir)
        print("ChromaDB client created, getting collection...")
        self.collection = client.get_collection(name=self.config.collection_name)
        print(f"Collection loaded: {self.collection.count()} chunks")
        
        print("Loading corpus pickle file …")
        try:
            with open(self.config.corpus_path, "rb") as f:
                print("Pickle file opened, loading...")
                self.corpus = pickle.load(f)
            print(f"Corpus loaded: {len(self.corpus)} entries")
        except Exception as e:
            print(f"Warning: Corpus not loaded ({e})")
            self.corpus = {}
        
        print("All models and data loaded. Ready!")
    
    def embed(self, text: str) -> list:
        out = self.model.encode(
            [text], 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        return out["dense_vecs"][0].tolist()
    
    def dense_search(self, query: str, top_k: Optional[int] = None) -> Tuple[list, list, list, list]:
        if top_k is None:
            top_k = self.config.dense_top_k
            
        res = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]
    
    def multivec_rerank(self, query: str, ids: list, docs: list) -> List[Tuple[str, float]]:
        if not self.corpus:
            return list(zip(ids, [0.0] * len(ids)))
        
        texts = [self.corpus.get(cid, doc) for cid, doc in zip(ids, docs)]
        pairs = [(query, t) for t in texts]
        scores = self.model.compute_score(
            sentence_pairs=pairs,
            weights_for_different_modes=list(self.config.fusion_weights),
            batch_size=16,
        )
        return sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    
    def crossencoder_rerank(
        self, 
        query: str, 
        ids: list, 
        docs: list, 
        top_n: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        if top_n is None:
            top_n = self.config.rerank_top_n
            
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        return sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:top_n]
    
    def search(self, query: str) -> str:
        """
        Execute full 3-stage retrieval pipeline:
        1. Dense search
        2. Multi-vector reranking
        3. Cross-encoder reranking
        """
        try:
            ids, dists, metas, docs = self.dense_search(query)
            
            mv_ranked = self.multivec_rerank(query, ids, docs)
            mv_ids = [cid for cid, _ in mv_ranked]
            
            mv_docs = [
                self.corpus.get(cid, docs[ids.index(cid)]) if cid in ids else "" 
                for cid in mv_ids[:self.config.dense_top_k]
            ]
            final = self.crossencoder_rerank(query, mv_ids, mv_docs)
            
            retrieved_contexts = []
            for rank, (cid, score) in enumerate(final, 1):
                doc_idx = mv_ids.index(cid)
                text_content = mv_docs[doc_idx]
                meta_idx = ids.index(cid) if cid in ids else 0
                meta = metas[meta_idx] if meta_idx < len(metas) else {}
                print(f"Sample metadata for doc {rank}:", meta)
                title = meta.get("title") or meta.get("source") or meta.get("url") or f"Document {rank}"
                retrieved_contexts.append(f"[{title}]:\n{text_content}")
                
            return "\n\n---\n\n".join(retrieved_contexts)
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"