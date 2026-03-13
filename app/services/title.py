"""
Support module for creating gen ai response , inserting it in DB and other stuff
"""
import os
import traceback
import oracledb


def create_new_chat_title(conn: oracledb.Connection,service, user_id: str, chat_id: str, user_query: str):
    """
    Create a new chat record using the provided service instance.

    This function extracts the core logic from the previous ChatService method and
    operates on the passed-in `service` (which must expose the same attributes as
    ChatService: llm_inference_client, llm_response_extractor, prompt_generator_client,
    sql_loader, adb_client).

    Args:
        service: An object exposing LLM clients and DB clients (typically ChatService).
        user_id: User identifier (if provided, persistence will be attempted).
        chat_id: Chat identifier.
        user_query: The user's initial message.
        app_state: Application state object to update in-memory mappings.

    Returns:
        title (str): Generated or fallback title for the chat.

    Raises:
        FileNotFoundError: If the prompt file is missing.
        Any exceptions from DB persistence are caught and logged but do not stop execution.
    """
    # Load prompt template from project prompts
    prompt_file = os.path.join("prompts", "create_chat_title.txt")
    with open(prompt_file, "r") as f:
        title_prompt_template = f.read()
    title_prompt = title_prompt_template.replace("{user_message}", user_query)

    # Call LLM
    raw_title = service.llm_inference_client.inference_single_input(user_query, title_prompt)
    if isinstance(raw_title, (tuple, list)):
        raw_title = raw_title[0]


    # Parse response
    try:
        service.llm_response_extractor.set_data(raw_title)
        title = service.llm_response_extractor.get("title", f"Chat_{chat_id[:8]}")
    except Exception:
        # Fall back to generated title
        traceback.print_exc()
        title = f"Chat_{chat_id[:8]}"

    # Persist to DB if requested
    if user_id:
        try:
            insert_user = service.sql_loader.insert_user_chat(user_id, chat_id, title)["insert_user_chat"]
            service.adb_client.execute_single_non_query(conn,insert_user)

        except Exception:
            traceback.print_exc()

    return user_id, chat_id,title
