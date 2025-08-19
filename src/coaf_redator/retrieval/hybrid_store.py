import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..utils.config import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HYBRID_ALPHA,
    TOP_K,
)


def extract_text(pdf_path: str) -> str:
    """Extrai texto do PDF com validação de existência."""
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"Norma 4001 não encontrada em: {p.resolve()}")
    reader = PdfReader(str(p))
    return "\n".join([(pg.extract_text() or "") for pg in reader.pages])


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Chunking simples com overlap por caracteres, preservando sentenças."""
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, curr, idx = [], "", 0
    for sent in sents:
        if len(curr) + len(sent) < chunk_size:
            curr += sent + " "
        else:
            chunks.append({"id": idx, "text": curr.strip()})
            idx += 1
            curr = (curr[-overlap:] if overlap < len(curr) else curr) + sent + " "
    if curr.strip():
        chunks.append({"id": idx, "text": curr.strip()})
    return chunks


@dataclass
class HybridStore:
    client: Any
    collection: Any
    embedder: SentenceTransformer
    bm25: BM25Okapi
    texts: List[str]

    @classmethod
    def from_pdf(cls, pdf_path: str):
        """
        Cria/abre a coleção do Chroma, indexa o PDF da 4001 com metadados,
        e constrói o índice BM25 em memória.
        """
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        texts = [c["text"] for c in chunks]
        ids = [str(c["id"]) for c in chunks]

        client = Client(
            Settings(persist_directory=CHROMA_DB_DIR, anonymized_telemetry=False)
        )
        embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        collection = client.get_or_create_collection(
            name="norma_4001",
            embedding_function=embed_func,
        )

        # Se a coleção estiver vazia, adiciona documentos com metadados (orig_id)
        if collection.count() == 0:
            metadatas = [{"orig_id": int(i)} for i in ids]
            collection.add(documents=texts, ids=ids, metadatas=metadatas)

        # Índice BM25 local
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)

        embedder = SentenceTransformer(EMBEDDING_MODEL)

        return cls(
            client=client,
            collection=collection,
            embedder=embedder,
            bm25=bm25,
            texts=texts,
        )

    def hybrid_query(
        self, query: str, top_k: int = TOP_K, alpha: float = HYBRID_ALPHA
    ) -> List[Dict[str, Any]]:
        """
        Combinação híbrida:
        - Similaridade de embeddings (Chroma)
        - BM25 (bag-of-words)
        alpha controla o peso da parte densa (embeddings).
        """
        start = time.time()

        # 🔁 CORRIGIDO: não pedir "ids" no include; usar "metadatas" e ler "orig_id"
        emb = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3,
            include=["documents", "distances", "metadatas"],
        )

        docs = emb.get("documents", [[]])[0]
        metas = emb.get("metadatas", [[]])[0] or []
        sims = [1 - d for d in emb.get("distances", [[]])[0]]

        # "ids" originais vêm de metadados ("orig_id"); se não existir, usa o índice local
        ids_from_meta = [m.get("orig_id", i) for i, m in enumerate(metas)]

        # BM25 normalizado
        bm_scores = self.bm25.get_scores(query.split())
        mn, mx = min(bm_scores), max(bm_scores) or 1.0
        bm_norm = [(s - mn) / (mx - mn) for s in bm_scores]

        # União de candidatos: top embeddings + top BM25
        cand = set(ids_from_meta)
        top_b = sorted(range(len(bm_norm)), key=lambda i: bm_norm[i], reverse=True)[
            : top_k * 3
        ]
        cand.update(top_b)

        # Acesso seguro às posições dos ids provenientes de embeddings
        id_pos = {i: pos for pos, i in enumerate(ids_from_meta)}

        ranked = []
        for i in cand:
            e = sims[id_pos[i]] if i in id_pos else 0.0
            b = bm_norm[i]
            ranked.append(
                {"id": i, "score": alpha * e + (1 - alpha) * b, "text": self.texts[i]}
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        took = time.time() - start
        return [
            {"rank": r + 1, **it, "elapsed_s": round(took, 2)}
            for r, it in enumerate(ranked[:top_k])
        ]
