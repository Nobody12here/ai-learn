FROM python:3.12-alpine
WORKDIR /app
COPY . .
RUN pip install uv
RUN uv sync
EXPOSE 8000
CMD [ "uv","run","uvicorn","src.api:app","--host","0.0.0.0","--port","8000" ]