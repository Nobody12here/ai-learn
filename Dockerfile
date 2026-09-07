FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv
RUN uv sync
COPY src ./src
COPY models ./models
EXPOSE 8000
CMD [ "uv","run","uvicorn","src.api:app","--host","0.0.0.0","--port","8000" ]