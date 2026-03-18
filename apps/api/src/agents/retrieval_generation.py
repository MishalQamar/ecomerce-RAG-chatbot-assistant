import openai
from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field
import instructor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np
from typing import Optional
from qdrant_client.http.models import Prefetch, Document, FusionQuery
from agents.prompts.utils.prompt_managment import prompt_template_config


openai_client = openai.OpenAI()
client = instructor.from_openai(openai_client)


class RAGUsedContext(BaseModel):
    id: str = Field(description="The id of the item used to answer the question")

    image_url: str = Field(
        description="The image url of the item used to answer the question"
    )
    price: Optional[float] = Field(
        description="The price of the item used to answer the question"
    )
    description: str = Field(
        description="Short description of the item used to answer the question"
    )


class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: list[RAGUsedContext] = Field(
        description="List of items used to answer the question"
    )


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embeddings(text, model="text-embedding-3-small"):
    response = openai_client.embeddings.create(
        input=text,
        model=model,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding


@traceable(name="retreive_data", run_type="retriever")
def retreive_data(query, qdrant_client, k=5):
    query_embedding = get_embeddings(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hydrid-search",
        prefetch=[
            Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings,
    }

    query_embedding = get_embeddings(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings,
    }


@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context):
    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += (
            f"Product ID: {id}\n,  Description: {chunk}\n,  Rating: {rating}\n\n"
        )

    return formatted_context


@traceable(name="build_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    template = prompt_template_config(
        "api/prompts/retrieval_generation.yaml", "retrieval_generation"
    )
    prompt = template.render(
        preprocessed_context=preprocessed_context, question=question
    )

    return prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"},
)
def generate_answer(prompt):
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=RAGGenerationResponse,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response


@traceable(name="rag_pipeline", run_type="chain")
def rag_pipeline(question, qdrant_client, k=5):

    retrieved_context = retreive_data(question, qdrant_client, k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    final_result = {
        "original_output": answer,
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"],
    }
    return final_result


def rag_pipeline_wrapper(question, k=5):
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = rag_pipeline(question, qdrant_client, k)
    used_context = []
    dummy_vector = np.zeros(1536).tolist()
    references = result.get("references") or []

    # Prefer explicit references from the LLM if present
    if references:
        ids_to_fetch = [(item.id, item.description) for item in references]
    else:
        # Fallback: use retrieved_context_ids and retrieved_context directly from retriever
        ids_to_fetch = list(
            zip(
                result.get("retrieved_context_ids", []),
                result.get("retrieved_context", []),
            )
        )

    for ref_id, ref_description in ids_to_fetch:
        payload = (
            qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hydrid-search",
                query=dummy_vector,
                limit=1,
                using="text-embedding-3-small",
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchValue(value=ref_id),
                        )
                    ]
                ),
            )
            .points[0]
            .payload
        )
        image_url = payload.get("image")
        price = payload.get("price")
        used_context.append(
            {
                "id": ref_id,
                "image_url": image_url,
                "price": price,
                "description": ref_description,
            }
        )
    return {
        "answer": result["answer"],
        "used_context": used_context,
    }
