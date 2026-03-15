from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
    Faithfulness,
    ResponseRelevancy,
)
from api.core.rag_pipeline import rag_pipeline

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)


async def ragas_faithfulness(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"],
    )
    scorer = Faithfulness(llm=ragas_llm)

    return await scorer.score_async(sample)


async def ragas_response_relevancy(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"],
    )
    scorer = ResponseRelevancy(llm=ragas_llm)

    return await scorer.score_async(sample)


async def ragas_id_based_context_precision(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"],
    )
    scorer = IDBasedContextPrecision()

    return await scorer.score_async(sample)


async def ragas_id_based_context_recall(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"],
    )
    scorer = IDBasedContextRecall()

    return await scorer.score_async(sample)


results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_id_based_context_precision,
        ragas_id_based_context_recall,
    ],
    experiment_prefix="retriever",
    max_concurrency=10,
)
