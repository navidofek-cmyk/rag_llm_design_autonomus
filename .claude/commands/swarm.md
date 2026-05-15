# /swarm

Swarm vyhledávání — více agentů prohledává nezávisle, bez koordinátora.

## Použití

```
/swarm "Co je GIL?"
/swarm "How does FastAPI handle dependencies?"
```

## Jak funguje

```
Spawn 3 swarm-agent najednou (v JEDNÉ zprávě):
  Agent 1 → python-docs  (rozhoduje sám)
  Agent 2 → peps         (rozhoduje sám)
  Agent 3 → fastapi      (rozhoduje sám)

Žádný koordinátor — každý agent je autonomní.

Po dokončení: sloučení výsledků přes RRF fusion.
Výsledek: odpověď s citacemi ze všech collections.
```

## Rozdíl od /ask

```
/ask    → jeden agent, jedna collection, koordinovaný retrieval
/swarm  → tři nezávislí agenti, tři collections, žádný koordinátor
```
