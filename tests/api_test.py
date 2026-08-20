from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

response = client.get("/")
print(response)