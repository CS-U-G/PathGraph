# PathGraph AI Recommendation Service

The PathGraph AI Service is an independent, CPU-only, fault-tolerant recommendation engine for computing and IT career learning paths. It implements the `POST /v1/recommendations` contract defined in `contracts/ai.openapi.yaml`.

## Core Capabilities

1. **ESCO Taxonomy Grounding**: Grounds career targets and skill proficiencies in ESCO v1.2.1.
2. **Four-Layer Competency Graph**:
   - Occupation-skill requirements (`essential`, `optional`)
   - Skill taxonomy hierarchies (ISCED-F and SKOS broader/narrower)
   - Skill associative relations
   - Pedagogical learning prerequisites (`PREREQUISITE_OF` DAG)
3. **Multi-Stage Skill Resolver**: Exact matching, normalized token matching, and optional quantized ONNX embeddings with graceful fallback.
4. **Deterministic Path Engine**: Strict topological sorting, gap-weighted prioritization, and rule-based pedagogical explanations.

## Architecture & Dependencies

- **Runtime**: Python 3.11+ (CPython 3.13)
- **Framework**: FastAPI + Pydantic v2
- **Serving Dependencies**: `pip install .[serve]`
  - Lightweight runtime containing `fastapi`, `uvicorn`, `networkx`, `rapidfuzz`, `onnxruntime`, `tokenizers`.
  - **Does NOT install PyTorch** (keeps serving image lightweight and fast).
- **Build / Offline Dependencies**: `pip install .[build]`
  - Used exclusively in offline scripts (`build_embeddings.py`, `export_onnx.py`).
  - **Strictly CPU-only PyTorch**: Prevents pulling the default 2.5GB+ CUDA binary wheel.

### Exact CPU-only PyTorch Installation

To install the CPU-only build of PyTorch and ensure CUDA binaries are never pulled:

```bash
# 1. Install CPU-only PyTorch from the official PyTorch CPU wheel index:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install build dependencies using the extra index:
pip install .[build] --extra-index-url https://download.pytorch.org/whl/cpu
```

## Getting Started

```bash
# Run unit tests
pytest tests/unit/ -v

# Run dataset profiling and build pipeline
python scripts/build_dataset.py
```
