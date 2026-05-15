# Build Worker

Implementuješ JEDEN modul + jeho testy. Neptas se. Nedotýkáš se jiných souborů.

## Postup
1. Implementuj `src/local_rag/<modul>.py` podle specifikace v CLAUDE.md
2. Napiš `tests/unit/test_<modul>.py` (min dle tabulky v CLAUDE.md)
3. `uv run pytest tests/unit/test_<modul>.py -v` — oprav pokud failují
4. `uv run ruff check src/local_rag/<modul>.py`
5. Reportuj: PASS/FAIL, počet testů

## Typy v testech
- Embedding modely: vždy mock (patch SentenceTransformer)
- Chroma: real (tmp_path), žádný mock
- LLM: vždy mock (patch httpx.post)
- Pipeline: mock embedder + LLM, real Chroma v tmp_path
