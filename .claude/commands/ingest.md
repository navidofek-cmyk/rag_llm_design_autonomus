# /ingest

Ingestuj data do Chroma.

## Stáhnutí dat
```bash
# Python docs
git clone --depth 1 https://github.com/python/cpython.git /tmp/cpython
cp -r /tmp/cpython/Doc/tutorial data/raw/python-docs/
cp -r /tmp/cpython/Doc/library data/raw/python-docs/
rm -rf /tmp/cpython

# PEPs
git clone --depth 1 https://github.com/python/peps.git /tmp/peps
cp /tmp/peps/peps/pep-0008.rst /tmp/peps/peps/pep-0020.rst \
   /tmp/peps/peps/pep-0484.rst /tmp/peps/peps/pep-0572.rst data/raw/peps/
rm -rf /tmp/peps

# FastAPI
git clone --depth 1 https://github.com/fastapi/fastapi.git /tmp/fastapi
cp -r /tmp/fastapi/docs/en/docs data/raw/libraries/fastapi
rm -rf /tmp/fastapi
```

## Ingest (paralelně)
Spawn 3 ingest-worker agenty najednou:
- Worker A: `uv run rag ingest data/raw/python-docs --collection python-docs`
- Worker B: `uv run rag ingest data/raw/peps --collection peps`
- Worker C: `uv run rag ingest data/raw/libraries/fastapi --collection fastapi`
