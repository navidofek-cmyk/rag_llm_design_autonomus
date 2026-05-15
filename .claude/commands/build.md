# /build

Postav celý RAG projekt od nuly. Bez ptání. Paralelně. Hned.

## Vrstvení agentů

```
Supervisor (watchdog, decision matrix)
  └── Orchestrator (koordinuje waves)
        ├── build-worker × 3  (Wave A — paralelně)
        ├── build-worker × 3  (Wave B — paralelně)
        ├── build-worker × 2  (Wave C — paralelně)
        ├── build-worker × 1  (Wave D)
        ├── build-worker × 2  (Wave E — paralelně)
        └── build-worker × 1  (Wave F)
```

## Spuštění

1. Aktivuj supervisor (@supervisor) — zůstane aktivní po celou dobu
2. Supervisor spustí orchestrator
3. Orchestrator spustí waves podle CLAUDE.md
4. Supervisor zachytí každou situaci kde by worker zastavil

Žádné otázky. Žádná potvrzení. Cokoliv nejasného → supervisor rozhodne → pokračuj.
