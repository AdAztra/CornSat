# CornSat

Research project on December corn futures and satellite imagery.

## Setup

```bash
uv sync
uv run pytest -q
```

## Data layout

- `data/raw/` - original downloads (read-only, never modify)
- `data/interim/` - intermediate ingest outputs (e.g. `modis/`)
- `data/processed/` - analysis-ready tables

## Configuration

Put `NASS_API_KEY` in a `.env` file (copy `.env.example`). Never commit `.env`.
