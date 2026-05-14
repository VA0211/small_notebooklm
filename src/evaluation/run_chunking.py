def _evaluate_strategy(
        strategy: ChunkingStrategy, output_dir: Path, test_cases: list[dict]
        ) -> dict[str, object]:
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=strategy.chunker)

    result_out: dict[str, object] = {
    "strategy_id": strategy.strategy_id,
    "chunk_count": chunk_count,
    "summary_metrics": {},
    }

    try:
        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider="vllm")
        df = result.to_pandas()
        result_out["summary_metrics"] = summary_metrics(df)

    except Exception as exc:
        result_out["error"] = str(exc)

    write_json(output_dir / f"{strategy.strategy_id}.json", result_out)
    return result_out