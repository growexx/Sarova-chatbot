"""
Chat API routes.

This module defines HTTP endpoints related to chat-based
Business Intelligence (BI) interactions for the Audit trails.
It acts as a thin controller layer that validates input schemas
and delegates business logic to the ChatService.
"""
from fastapi import APIRouter
from app.schemas.chat import ChatInquiryRequest , ChatLoadRequest , LoadChatsPreviewRequest ,SignoutRequest, ChatDeletionRequest , AllChatDeletionRequest
from app.services.chat_service import ChatService
from fastapi import Request


# ---------------------------------------------------------
# Router initialization for chat-related endpoints
# ---------------------------------------------------------

router = APIRouter()
service = ChatService()

@router.post("/chat-inquiry")
async def chat_inquiry(
    request: ChatInquiryRequest,
    req: Request
    ):
    """
    Handle a chat inquiry from the user.

    This endpoint receives a user query along with a chat identifier,
    forwards the request to the ChatService for processing, and returns
    a structured response containing the query result, status, and
    any generated insights or errors.

    Args:
        request (ChatInquiryRequest): Validated request body containing
            chat_id and user_message.
        req (Request): FastAPI request object used to access
            application-level state (chat history, last SQL query, etc.).

    Returns:
        dict: A response dictionary produced by ChatService, including:
            - status code
            - LLM response
            - optional SQL query or data artifacts
    """
    return await service.handle_inquiry(
        request.user_id,
        request.chat_id,
        request.user_message,
        req.app.state
    )

@router.post("/load-historical-chat")
def load_chat_history(
    request: ChatLoadRequest,
    req: Request
    ):
    """
    Handle a chat load request from the user.

    This endpoint receives a user query along with a chat identifier,
    forwards the request to the ChatService for processing, and returns
    a structured response containing the chat history corresponding to chat_id and user_id

    Args:
        request (ChatLoadRequest): Validated request body containing
            chat_id and user_message.
        req (Request): FastAPI request object used to access
            application-level state (chat history, last SQL query, etc.).

    Returns:
        dict: A response dictionary produced by ChatService, including:
            - status code
            - chat_history
    """

    return service.load_chat_history(
        request.user_id,
        request.chat_id,
        req.app.state
    )

@router.post("/load-chats-preview")
def load_chat_previews(
    request: LoadChatsPreviewRequest,
    req: Request
    ):
    """
    Handle a loading of user chats from the user.

    This endpoint receives a user id,
    forwards the request to the ChatService for processing, and returns
    a structured response containing the chat ids, status.

    Args:
        request (LoadChatsPreviewRequest): Validated request body containing
            chat_id .
        req (Request): FastAPI request object used to access
            application-level state (chat history, last SQL query, etc.).

    Returns:
        dict: A response dictionary produced by ChatService, including:
            - status code
            - previous chat previews
    """
    return service.load_user_chats_previews(
        request.user_id,
        req.app.state
    )

@router.post("/user-signout")
def signout_processes(
    request: SignoutRequest,
    req: Request
    ):
    """
    Handle a Signout from the user.

    This endpoint receives a signout from user,
    forwards the request to the ChatService for deleting all run time memory for user

    Args:
        request (SignoutRequest): Validated request body containing
            chat_id.
        req (Request): FastAPI request object used to access
            application-level state (chat history, last SQL query, etc.).

    Returns:
        dict: A response dictionary produced by ChatService, including:
            - status code
    """
    return service.chat_runtime_cleanup(
        request.user_id,
        req.app.state
    )

@router.delete("/delete-chats")
def delete_chats(
    request: ChatDeletionRequest,
    req: Request
    ):
    """
    Handle a chat Deletion request from the user.

    This endpoint receives a deletion from user,
    forwards the request to the ChatService for deleting all chat_history for user

    Args:
        request (SignoutRequest): Validated request body containing
            user_id ,chat_ids .
        req (Request): FastAPI request object used to access
            application-level state (chat history, last SQL query, etc.).

    Returns:
        dict: A response dictionary produced by ChatService, including:
            - status code
    """
    return service.delete_chat_history(
        request.user_id,
        request.chat_ids,
        req.app.state
    )

@router.delete("/delete-all_chats")
def delete_all_chats(
    request: AllChatDeletionRequest,
    req: Request
    ):
    """
    Handle a chat Deletion request from the user.

    This endpoint receives a deletion from user,
    forwards the request to the ChatService for deleting all chat_history for user

    Args:
        request (SignoutRequest): Validated request body containing
            user_id ,chat_ids .
        req (Request): FastAPI request object used to access
            application-level state (chat history, last SQL query, etc.).

    Returns:
        dict: A response dictionary produced by ChatService, including:
            - status code
    """
    return service.delete_all_chats_for_user(
        request.user_id,
        req.app.state
    )

@router.get("/view-state")
def view_state(req: Request):
    """Simple api for viewing state, for testing purpose only"""
    return req.app.state
