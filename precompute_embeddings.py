import json
import numpy as np
from fastembed import TextEmbedding

CATALOG_PATH    = "data/shl_product_catalog.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
MODEL_NAME      = "BAAI/bge-small-en-v1.5"


def build_embedding_text(item: dict) -> str:
    parts = []
    if item.get("name"):
        parts.append(f"Assessment: {item['name']}")
    if item.get("description"):
        parts.append(f"Description: {item['description']}")
    if item.get("keys"):
        parts.append(f"Categories: {', '.join(item['keys'])}")
    if item.get("job_levels"):
        parts.append(f"Job levels: {', '.join(item['job_levels'])}")
    if item.get("languages"):
        parts.append(f"Languages: {', '.join(item['languages'][:5])}")
    if item.get("duration"):
        parts.append(f"Duration: {item['duration']}")
    return " | ".join(parts)


def main():
    print(f"Loading catalog from {CATALOG_PATH}...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} items loaded.")

    texts = [build_embedding_text(item) for item in catalog]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = TextEmbedding(MODEL_NAME)

    print("Encoding all catalog items (this takes ~30 seconds)...")
    embeddings = np.array(list(model.embed(texts)), dtype="float32")

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {EMBEDDINGS_PATH}")
    print(f"File size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    print()
   


if __name__ == "__main__":
    main()