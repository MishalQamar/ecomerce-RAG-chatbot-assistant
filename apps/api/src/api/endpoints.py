import logging
from fastapi import Request, APIRouter
from api.models import RagRequest, RagResponse
from agents.retrieval_generation import rag_pipeline

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: RagRequest) -> RagResponse:

    answer = rag_pipeline(payload.query)
    return RagResponse(request_id=request.state.request_id, answer=answer)


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
