from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Meeting RAG API",
    description="FastAPI interface for the Meeting RAG system.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Meeting RAG API is running."
    }