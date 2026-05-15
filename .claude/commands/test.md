# /test

Spusť testy. Neptas se. Reportuj výsledek.

```bash
uv run pytest tests/unit -q
uv run ruff check src/
uv run mypy src/local_rag
```

Výsledek: PASS (X/X) nebo FAIL (seznam failing testů).
