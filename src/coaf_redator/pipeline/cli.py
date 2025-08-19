import argparse
import json
from pathlib import Path

from ..utils.config import NORM_4001_PDF
from ..retrieval.hybrid_store import HybridStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Roda um teste simples no RAG híbrido"
    )
    parser.add_argument(
        "--q",
        type=str,
        default="fragmentação de depósitos em espécie",
        help="Consulta de teste",
    )
    args = parser.parse_args()

    if args.smoke_test:
        # Log do caminho da norma pra ajudar debug
        print("Usando norma em:", Path(NORM_4001_PDF).resolve())
        try:
            store = HybridStore.from_pdf(NORM_4001_PDF)
            hits = store.hybrid_query(args.q, top_k=5)
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        except FileNotFoundError as e:
            print(f"[ERRO] {e}")
            print(
                "Dica: verifique se o PDF existe no caminho acima ou ajuste NORM_4001_PDF no .env"
            )
        except Exception as e:
            print("[ERRO] Falha no smoke-test:", repr(e))
            raise
        return


if __name__ == "__main__":
    main()
