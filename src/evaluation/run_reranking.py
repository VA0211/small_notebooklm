from sentence_transformers import CrossEncoder
from src.rag import ANSWER_TEMPLATE, format_citations, render_prompt, retrieve
from src.schemas import RagAnswer
from src.llm import invoke_llm

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

def answer_with_reranker(
    question: str,
    collection_name: str,
    reranker: CrossEncoder,
    initial_k: int = 15,
    rerank_k: int = 5,
    filters: dict[str, object] | None = None,
    ) -> RagAnswer:
    # Giai đoạn 1: Truy xuất thô (initial_k)
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)

    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.",
            )

    # Giai đoạn 2: Tính toán điểm số tương quan chéo bằng Cross-Encoder
    scores = reranker.predict([[question, chunk.text] for chunk in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    # Xếp hạng lại và lọc ra các đoạn liên quan nhất (rerank_k)
    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]

    # Đưa ngữ cảnh đã được lọc vào prompt cho LLM
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked)
    text = invoke_llm(prompt)
    
    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(reranked),
        chunks=reranked,
        )