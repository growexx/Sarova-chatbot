"""
Schema for chat inquiry requests
"""
from pydantic import BaseModel

class ChatInquiryRequest(BaseModel):
    """
    Chat Inquiry Request only requires user input and Chat id, We are fixing same format
    """
    user_id: str
    chat_id: str
    user_message: str

class  ChatLoadRequest(BaseModel):
    """
    Chat Load Request only requires user input and Chat id.
    """
    user_id: str
    chat_id: str

class LoadChatsPreviewRequest(BaseModel):
    """
    Load Chat Previes for Front End
    """
    user_id: str

class SignoutRequest(BaseModel):
    """
    Signout Request only requires user input and Chat id.
    """
    user_id: str

class ChatDeletionRequest(BaseModel):
    """
    Chat Deletion Request only requires user input and Chat id.
    """
    user_id: str
    chat_ids: list

class AllChatDeletionRequest(BaseModel):
    """
    Chat Deletion Request only requires user input and Chat id.
    """
    user_id: str
