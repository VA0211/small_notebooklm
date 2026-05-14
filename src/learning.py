import json

from loguru import logger
from pydantic import ValidationError

from src.config import settings
from src.llm import invoke_llm
from src.rag import fetch_all_chunks, format_citations, render_prompt, retrieve
from src.schemas import Flashcard, FlashcardSet, QuizItem, QuizSet, RetrievedChunk, Summary

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"

def _parse_json(text):
    """Parse JSON object/array from model output, allowing optional markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].removesuffix("```").strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from model output: {cleaned}") from e

    if not isinstance(obj, (dict, list)):
        raise RuntimeError(f"Expected JSON object or array, got {type(obj).__name__}.")

    return obj

def _validate_items(
    payload: object,
    key: str,
    model_class: type,
    dedup_field: str,
    label: str,
    valid_markers: set[str],
) -> list:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object for {label}.")
    raw_items = payload.get(key)
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Expected '{key}' to be a list for {label}.")

    items: list = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            item = model_class.model_validate(raw)
        except ValidationError as e:
            logger.warning("Dropping invalid {}: {}", label, e)
            continue
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))

    if not items:
        raise RuntimeError(f"No valid {label} produced.")
    return items


def _validate_summary_payload(payload: object) -> tuple[str, list[str]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object for summary.")
    summary = payload.get("summary")
    key_points = payload.get("key_points", [])
    if not isinstance(summary, str):
        raise RuntimeError("Summary payload missing 'summary' string.")
    if not isinstance(key_points, list) or not all(isinstance(x, str) for x in key_points):
        raise RuntimeError("Summary payload 'key_points' must be a list of strings.")
    return summary.strip(), [kp.strip() for kp in key_points if kp.strip()]

def _resolve_target(document, query, filters, k, retrieval_k):
    effective_filters = dict(filters or {})

    if document:
        effective_filters["filename"] = document
    
    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective_filters)
        return chunks, "query", query
    
    if effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        scope = "document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in effective_filters.items())
        return chunks, scope, target
    
    return fetch_all_chunks(filters=None), "corpus", None

def summarize(document=None, query=None, filters=None, k=None):
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.summarize_retrieval_k
    )

    if len(chunks) <= settings.summarize_batch_size:
        prompt = render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks)
        payload = _parse_json(invoke_llm(prompt))
        summary_text, key_points = _validate_summary_payload(payload)
    else:
        partials = []
        for start in range(0, len(chunks), settings.summarize_batch_size):
            batch = chunks[start : start + settings.summarize_batch_size]
            payload = _parse_json(invoke_llm(render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch)))

            summary_text, key_points = _validate_summary_payload(payload)
            partials.append({"summary": summary_text, "keypoints": key_points})

        payload = _parse_json(invoke_llm(render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials)))
        summary_text, key_points = _validate_summary_payload(payload)

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks),
        chunks=chunks,
    )

def generate_quiz(document=None, query=None, filters=None, count=None, k=None):
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )

    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))

    items = _validate_items(payload, "items", QuizItem, "question", "quiz items", valid_markers)

    return QuizSet(scope=scope, target=target, items=items, chunks=chunks, citations=format_citations(chunks))

def generate_flashcards(document=None, query=None, filters=None, count=None, k=None):
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    n = count or settings.flashcards_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))

    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers)

    return Flashcard(
        scope=scope,
        target=target,
        cards=cards,
        chunks=chunks,
        citation=format_citations(chunks)
    )

