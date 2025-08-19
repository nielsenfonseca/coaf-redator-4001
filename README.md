# coaf-redator-4001

Redator de comunicações de suspeita de lavagem de dinheiro (COAF), com pipeline de agentes (LangGraph + LangChain) e RAG híbrido (ChromaDB + BM25) para:
- Ler dados cadastrais/transacionais;
- Gerar indicadores (incompatibilidade com renda, fracionamento ~2k/10k/50k, depósitos em espécie, contrapartes PEP etc.);
- Sugerir enquadramentos com base na **Carta Circular nº 4.001 (BCB)**;
- Produzir JSON estruturado com flags, textos e trilha de curadoria.

## Stack
- Python 3.11+, LangGraph, LangChain (Gemini), ChromaDB, rank-bm25, sentence-transformers.

## Como rodar
1. Crie a venv e instale `requirements.txt`
2. Copie `.env.example` para `.env` e preencha `GEMINI_API_KEY`
3. Baixe a norma 4001 em `docs/normativos/`
4. Rode `python -m src.coaf_redator.pipeline.cli --smoke-test`

## Avisos
Não suba dados reais/sigilosos. Use dados sintéticos localmente.
