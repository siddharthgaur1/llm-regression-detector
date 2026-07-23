FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root: the runner reads prompts/golden data and writes under data/runs.
RUN useradd --uid 10001 evaluator && chown -R evaluator:evaluator /app
USER evaluator

ENTRYPOINT ["python", "-m", "src.runner"]
