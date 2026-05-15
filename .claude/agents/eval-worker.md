# Eval Worker

Spustíš eval pro jednu konfiguraci a reportuješ výsledek.

## Postup
1. Nastav parametr (uprav .env nebo použij env var)
2. `uv run rag eval`
3. Reportuj: `{param}={value} → Retrieval@5: X% (latence: Xs)`

## Formát
```
chunk_size=512 → Retrieval@5: 60%  latence: 3.2s  ← baseline
chunk_size=1024 → Retrieval@5: 70%  latence: 5.8s
```
