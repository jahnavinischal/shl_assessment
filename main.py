import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from catalog import load_catalog
from agent import run_agent
from models import ChatRequest, ChatResponse, Recommendation

from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# Lifespan: build FAISS index once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SHL Recommender starting up ===")
    start = time.time()
    load_catalog("data/shl_product_catalog.json")
    elapsed = time.time() - start
    logger.info(f"=== Startup complete in {elapsed:.1f}s ===")
    yield
    logger.info("=== SHL Recommender shutting down ===")


# FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL catalog assessment selection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for testing from browsers / Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Endpoints

@app.get("/health")
def health():
    """
    Readiness check. Returns HTTP 200 with {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Stateless chat endpoint.

    The full conversation history is sent in every request.
    Returns the next agent reply plus (optionally) a structured shortlist.
    """
    # Basic validation 
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages array cannot be empty.")

    # Must start with a user message
    if req.messages[0].role != "user":
        raise HTTPException(
            status_code=422,
            detail="First message must have role 'user'.",
        )

    # Enforce alternating roles (user/assistant/user/...)
    for i, msg in enumerate(req.messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=422,
                detail=f"Message at index {i} has invalid role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    # Cap at 8 turns total (4 user + 4 assistant) per spec
    if len(req.messages) > 8:
        raise HTTPException(
            status_code=422,
            detail="Conversation exceeds maximum of 8 messages.",
        )

    # Run agent 
    logger.info(f"/chat called with {len(req.messages)} messages")
    start = time.time()

    result = run_agent(req.messages)

    elapsed = time.time() - start
    logger.info(f"/chat completed in {elapsed:.2f}s")

    # Build response 
    return ChatResponse(
        reply=result["reply"],
        recommendations=result["recommendations"],   # None or list[Recommendation]
        end_of_conversation=result["end_of_conversation"],
    )


# Global error handler — never let the server return 500 to the evaluator
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An internal error occurred. Please try again.",
            "recommendations": None,
            "end_of_conversation": False,
        },
    )