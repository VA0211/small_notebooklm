from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


_RECURSIVE_CONFIGS = [("rc_500_50", 500, 50),
                    ("rc_800_100", 800, 100),
                    ("rc_1000_150", 1000, 150),
                    ("rc_1500_200", 1500, 200),
                    ]

@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: object
    params: dict[str, object]
@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] | None = None
def _splitter(self) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators or DEFAULT_SEPARATORS,
                is_separator_regex=False,
            )
def split_documents(self, documents: list[Document]) -> list[Document]:
    if not documents:
        return []
    return self._splitter().split_documents(documents)


_SEMANTIC_CONFIGS = [("semantic_percentile", "percentile"),
                    ("semantic_std_dev", "standard_deviation"),
                    ("semantic_interquartile", "interquartile"),
                    ]

@dataclass(frozen=True)
class SemanticChunkerWrapper:
    """Wrapper for LangChain SemanticChunker."""
    embeddings: Embeddings
    breakpoint_type: str = "percentile"

def _splitter(self) -> SemanticChunker:
    return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_type,
            )

def split_documents(self, documents: list[Document]) -> list[Document]:
    if not documents:
        return []
    return self._splitter().split_documents(documents)

def split_text(self, text: str) -> list[str]:
    return self._splitter().split_text(text)