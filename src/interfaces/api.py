from pathlib import Path
from pydantic import BaseModel, Field
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from fastapi import FastAPI

from indexing import save_and_ingest_pdf
from filters import filters_to_dict

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None

class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)

class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)

class FlashcardsRequest(QuizRequest):
    pass

api = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0"
)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return save_and_ingest_pdf(content, file.filename or "")

@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    return answer(req.question, k=req.k, filters=filters_to_dict(req.filters))

@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest):
    return summarize_learning(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        k=req.k,
    )

