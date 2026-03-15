import logging
from fastapi import Request, APIRouter
from api.models import RagRequest, RagResponse
from agents.retrieval_generation import rag_pipeline_wrapper
from qdrant_client import QdrantClient
from agents.retrieval_generation import RAGUsedContext

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")

rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: RagRequest) -> RagResponse:

    result = rag_pipeline_wrapper(payload.query, k=5)
    return RagResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_context=[RAGUsedContext(**(item)) for item in result["used_context"]],
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
