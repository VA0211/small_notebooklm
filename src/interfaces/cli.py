@app.command()
def ingest(recreate=False):
    count = ingest_data_dir(recreate=recreate)
    typer.echo(f"Done. {count} chunks indexed.")

@app.command()
def ask(question, k=None, filters=None):
    result = answer(question, k=k, filters= _parse_filters(filters))
    _print_answer(result.answer)
    _print_sources(result.chunks)
    
@app.command("debug-retrieval")
def debug_retrieval(question, k=None, filters=None, as_json=False):
    chunks = retrieve(question, k=k, filters=_parse_filters(filters))
    typer.echo(json.dump([c.model_dump() for c in chunks], 
                         ensure_ascii=False, indent=2))
    
@app.command("summarize")
def summarize(document=None, query=None, filters=None, k=None, output=None, fmt="text"):
    result = summarize_learning(document=document, query=query, filters=_parse_filters(filters), k=k)
    _emit(result, output, fmt)

    