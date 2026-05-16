import json
import os
import faiss
from sentence_transformers import SentenceTransformer

# Globals (populated once by load_catalog() at app startup)
_catalog: list[dict] = []
_index: faiss.IndexFlatIP | None = None
_model: SentenceTransformer | None = None

# Map catalog keys field values → single-letter test_type codes
KEY_TO_TYPE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Simulations": "S",
}


def _get_test_type(keys: list[str]) -> str:
    """
    Return combined type codes for all keys an item has.
    e.g. ["Knowledge & Skills", "Simulations"] -> "K,S"
    e.g. ["Personality & Behavior", "Competencies"] -> "P,C"
    Falls back to "A" if no keys match.
    """
    codes = []
    seen = set()
    for key in keys:
        t = KEY_TO_TYPE.get(key)
        if t and t not in seen:
            codes.append(t)
            seen.add(t)
    return ",".join(codes) if codes else "A"


def _build_embedding_text(item: dict) -> str:
    """
    Build a rich text representation of a catalog item for embedding.
    More detail = better retrieval.
    """
    parts = []

    name = item.get("name", "")
    if name:
        parts.append(f"Assessment: {name}")

    description = item.get("description", "")
    if description:
        parts.append(f"Description: {description}")

    keys = item.get("keys", [])
    if keys:
        parts.append(f"Categories: {', '.join(keys)}")

    job_levels = item.get("job_levels", [])
    if job_levels:
        parts.append(f"Job levels: {', '.join(job_levels)}")

    languages = item.get("languages", [])
    if languages:
        parts.append(f"Languages: {', '.join(languages[:5])}")

    duration = item.get("duration", "")
    if duration:
        parts.append(f"Duration: {duration}")

    remote = item.get("remote", "")
    if remote:
        parts.append(f"Remote testing: {remote}")

    adaptive = item.get("adaptive", "")
    if adaptive:
        parts.append(f"Adaptive: {adaptive}")

    return " | ".join(parts)


def load_catalog(path: str = "data/catalog.json") -> None:
    """
    Load catalog JSON, embed every item, build FAISS index.
    Called once at app startup via FastAPI lifespan.
    """
    global _catalog, _index, _model

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalog not found at '{path}'. "
            "Place your catalog.json inside the data/ directory."
        )

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Filter to Individual Test Solutions only (exclude pre-packaged job solutions)
    # The catalog JSON already contains only individual test solutions per the task but we defensively keep all items since the JSON is pre-filtered.
    _catalog = raw

    print(f"[catalog] Loaded {len(_catalog)} items from {path}")

    # Load embedding model (downloads ~90MB on first run, cached after)
    print("[catalog] Loading sentence-transformer model...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build embedding texts
    texts = [_build_embedding_text(item) for item in _catalog]

    # Encode all items (batched, with progress)
    print("[catalog] Encoding catalog items...")
    embeddings = _model.encode(
        texts,
        normalize_embeddings=True,  # L2 normalise for cosine via inner product
        batch_size=64,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype("float32")

    # Build flat inner-product index (exact search, fine for <10k items)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings)

    print(f"[catalog] FAISS index built: {_index.ntotal} vectors, dim={dim}")


def search(query: str, k: int = 15) -> list[dict]:
    """
    Semantic search over the catalog.
    Returns up to k items with full metadata + similarity score.
    """
    if _model is None or _index is None:
        raise RuntimeError("Catalog not loaded. Call load_catalog() first.")

    q_emb = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = _index.search(q_emb, min(k, _index.ntotal))

    results = []
    for idx, score in zip(idxs[0], scores[0]):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        item = _catalog[idx]
        results.append({
            "name": item.get("name", ""),
            "url": item.get("link", ""),
            "test_type": _get_test_type(item.get("keys", [])),
            "description": item.get("description", ""),
            "keys": item.get("keys", []),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", ""),
            "duration_raw": item.get("duration_raw", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
            "score": float(score),
        })

    return results


def get_item_by_name(name: str) -> dict | None:
    """Exact name lookup (case-insensitive) for comparison queries."""
    name_lower = name.lower().strip()
    for item in _catalog:
        if item.get("name", "").lower().strip() == name_lower:
            return item
    return None


def get_all_items() -> list[dict]:
    """Return full catalog — used for building system context."""
    return _catalog