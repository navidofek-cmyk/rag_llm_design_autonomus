# /experiment

Parallel parameter sweep.

## Použití
```
/experiment chunk_size 256,512,1024,2048
/experiment top_k 3,5,10,20
```

## Postup
1. Spawn max 3 eval-worker agenty najednou
2. Každý testuje jednu hodnotu
3. Mezi vlnami: 30s (Ollama rate limit)
4. Výsledky → tabulka → zapiš do docs/experiments.md
