import logging
import json
from fastapi import Request, APIRouter
from fastapi.responses import StreamingResponse
from api.models import RagRequest, RagResponse
from agents.retrieval_generation import rag_pipeline_wrapper, rag_pipeline_stream_wrapper
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


@rag_router.post("/stream")
def rag_stream(request: Request, payload: RagRequest) -> StreamingResponse:
    result = rag_pipeline_stream_wrapper(payload.query, k=5)
    answer_stream = result["answer_stream"]
    used_context = result["used_context"]

    def event_generator():
        for token in answer_stream:
            yield json.dumps({"type": "token", "content": token}) + "\n"
        yield (
            json.dumps(
                {
                    "type": "done",
                    "request_id": request.state.request_id,
                    "used_context": used_context,
                }
            )
            + "\n"
        )

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
