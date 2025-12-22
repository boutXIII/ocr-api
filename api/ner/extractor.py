# =========================================
# Imports
# =========================================
from api.ner.strategy import DOCUMENT_STRATEGY
from api.utils.documents import run_gliner

# =========================================
# Entity extraction
# =========================================
def extract_entities(
    *,
    text: str,
    document_class: str,
    gliner_threshold: float,
) -> list[dict]:

    strategy = DOCUMENT_STRATEGY.get(document_class)
    if not strategy:
        raise ValueError(f"Unknown document class: {document_class}")

    entities: list[dict] = []

    # ---------------- GLiNER ----------------
    gliner_labels = [
        label for label, cfg in strategy.items()
        if cfg["extractor"] == "gliner"
    ]

    if gliner_labels:
        gliner_entities = run_gliner(
            text=text,
            document_class=document_class,
            threshold=gliner_threshold,
        )

        for e in gliner_entities:
            if e["label"] in gliner_labels:
                entities.append(e)

    # ---------------- REGEX ----------------
    for label, cfg in strategy.items():
        if cfg["extractor"] != "regex":
            continue

        pattern = cfg["pattern"]
        score = cfg.get("score", 1.0)

        for m in pattern.finditer(text):
            entities.append({
                "label": label,
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "score": score,
            })

    return entities
