import openai
from qdrant_client import QdrantClient


def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


def retreive_data(query, qdrant_client, k=5):
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


def build_prompt(preprocessed_context, question):
    prompt = f"""
    You are a helpful shopping assistant that can answer questions about the product descriptions.You will be given a question and a list of product descriptions.
    Your task is to answer the question based on the product descriptions.
    context: {preprocessed_context}
    Question: {question}
    """
    return prompt


def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="minimal",
    )
    return response.choices[0].message.content


def rag_pipeline(question, k=5):
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    retrieved_context = retreive_data(question, qdrant_client, k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    return answer
