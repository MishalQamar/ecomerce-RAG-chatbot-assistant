run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-eval-retriever:
	uv sync
	PYTHONPATH=${PWD}/apps/api/src:${PWD}/apps/api uv run --env-file .env apps/api/evals/eval_retriever.py