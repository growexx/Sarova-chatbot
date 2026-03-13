"""
Application entry point for the Sarova Extended Chatbot API.

This module initializes the FastAPI application, configures global
middleware such as CORS, sets up shared application state, and
registers all API route modules.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for FastAPI.

    This function manages application-level startup and shutdown events.
    It ensures that shared infrastructure resources are properly
    initialised before the application begins handling requests and
    cleanly released when the application shuts down.

    Startup:
        - Initializes the Oracle Autonomous Database (ADB) connection pool
          via the shared ChatService instance.
        - Pre-warms database connections to avoid first-request latency.

    Runtime:
        - The application serves requests while the lifespan context
          remains active.

    Shutdown:
        - Closes and drains the ADB connection pool.
        - Ensures all active database connections are released cleanly.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded back to FastAPI while the application
        is running.

    Notes:
        - The connection pool is created only once at startup.
        - Proper cleanup prevents connection leaks and ensures graceful
          application shutdown.
        - This function should be registered with FastAPI using the
          `lifespan` parameter during app initialization.
    """
    # ── Startup ────────────────────────────────────────────
    print("[Lifespan] Starting up — initialising DB connection pool ...")
    # chat.service is the single ChatService instance created in chat.py
    chat.service.adb_client.init_pool()
    print("[Lifespan] DB connection pool ready")

    yield

    # ── Shutdown ───────────────────────────────────────────
    print("[Lifespan] Shutting down — closing DB connection pool ...")
    chat.service.adb_client.close_pool()
    print("[Lifespan] DB connection pool closed")

# ---------------------------------------------------------
# Create FastAPI application instance
# ---------------------------------------------------------

app = FastAPI(
    title="Sarova Extented Bot Web API",
    lifespan=lifespan
)
# ---------------------------------------------------------
# Configure CORS middleware
# Allows cross-origin requests for frontend integration
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.last_user_chat = {}
# user id key and active chat id
app.state.chat_history = {}
# chat id and respective chat_history
app.state.last_sql_queries = {}
# last chat id and sql query

# ---------------------------------------------------------
# Register API routers
# ---------------------------------------------------------
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

