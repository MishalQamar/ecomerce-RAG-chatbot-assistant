from pydantic import BaseModel, Field
from agents.retrieval_generation import RAGUsedContext


class RagRequest(BaseModel):
    query: str = Field(..., description="The query to be used in RAG pipeline")


class RagResponse(BaseModel):
    request_id: str = Field(..., description="The request id")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContext] = Field(..., description="The used context")
