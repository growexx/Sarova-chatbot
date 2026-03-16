"""
OCI Generative AI Chat Inference Module - Compatible with main6.py
This is an improved version of oci_llm.py that works seamlessly with the simplified chatbot API.
Key improvements:
- Direct compatibility with main6.py chat history format
- Automatic message format conversion
- Simplified interface
- Better error handling
- Support for different role types (USER, ASSISTANT, SYSTEM)
Dependencies:
- OCI SDK for Python
"""
import traceback
import oci
from oci.generative_ai_inference.models import GenericChatRequest, TextContent, Message
from typing import List, Dict, Optional
import logging
from oci.exceptions import RequestException, ServiceError
# Set up logging
import configparser
logger = logging.getLogger(__name__)

# app/exceptions/llm_exceptions.py

class LLMInferenceError(RuntimeError):
    """Raised when LLM inference fails."""


class LLMInference:
    """
    Enhanced OCI Generative AI interface compatible with main6.py chatbot.
    This class provides a simplified interface to Oracle Cloud Infrastructure's
    Generative AI service, with automatic message format conversion and
    better integration with the chatbot's session management.
    """

    def __init__(self, config_file: str = 'config.ini', model_id: Optional[str] = None):
        """
        Initialize the OCI LLM client.
        Args:
            config_file (str): Path to OCI configuration file
            model_id (str, optional): Override model ID from config. If None, uses GENAI.model_id.
        """
        try:
            # OCI Configuration
            genai_config = configparser.ConfigParser()
            genai_config.read(config_file)
            
            self.compartment_id = genai_config['GENAI']['compartment_id']
            self.CONFIG_PROFILE = "DEFAULT"
            self.config = oci.config.from_file(config_file, self.CONFIG_PROFILE)
            # Service endpoint
            self.endpoint = genai_config['GENAI']['endpoint']
            # Initialize the client
            self.generative_ai_inference_client = (
                oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=self.config,
                    service_endpoint=self.endpoint,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                    timeout=(5, 30)
                )
            )
            # Model: use override if provided, else config
            if model_id is None:
                model_id = genai_config["GENAI"]["model_id"].strip()
            # Chat configuration
            self.chat_detail = oci.generative_ai_inference.models.ChatDetails(
                compartment_id=self.compartment_id,
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=model_id
                )
            )
            logger.info("OCI LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCI LLM client: {str(e)}")
            raise

    @staticmethod
    def _convert_message_to_oci_format(role: str, message: str) -> Message:
        """
        Convert a simple message to OCI Message format.
        Args:
            role (str): Message role (USER, ASSISTANT, SYSTEM)
            message (str): Message content
        Returns:
            Message: OCI formatted message object
        """
        # Map role names to OCI format
        role_mapping = {
            "USER": "USER",
            "ASSISTANT": "ASSISTANT",
            "SYSTEM": "SYSTEM"
        }
        oci_role = role_mapping.get(role.upper(), "USER")

        content = TextContent()
        content.text = message
        content.type = 'TEXT'

        oci_message = Message()
        oci_message.role = oci_role
        oci_message.content = [content]
        return oci_message

    def _convert_chat_history_to_oci_format(self, chat_history: List[Dict[str, str]]) -> List[Message]:
        """
        Convert main6.py chat history format to OCI Message format.
        Args:
            chat_history (List[Dict]): Chat history from main6.py
                Format: [{"role": "USER", "message": "Hello", "timestamp": "..."}]
        Returns:
            List[Message]: OCI formatted messages
        """
        try:
            oci_messages = []
            for msg in chat_history:
                role = msg.get("role", "USER")
                message = msg.get("message", "")
                # Skip empty messages
                if not message.strip():
                    continue
                oci_message = self._convert_message_to_oci_format(role, message)
                oci_messages.append(oci_message)
            return oci_messages
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to convert chat history to OCI format: {str(e)}")
            raise

    def inference_from_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Generate LLM response from chat history (main6.py compatible).
        Args:
            chat_history (List[Dict]): Chat history from main6.py session
                Format: [{"role": "USER", "message": "Hello", "timestamp": "..."}]
        Returns:
            str: Generated response from the LLM
        Raises:
            Exception: If inference fails
        """
        try:
            # Convert chat history to OCI format
            oci_messages = self._convert_chat_history_to_oci_format(chat_history)
            if not oci_messages:
                logger.warning("No valid messages found in chat history")
                return "I'm sorry, I didn't receive any valid messages to respond to."

            chat_request = GenericChatRequest(
                api_format=GenericChatRequest.API_FORMAT_GENERIC,
                messages=oci_messages,
                max_tokens=6000,
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=0.75
            )
            # Set the request and make the call
            self.chat_detail.chat_request = chat_request
            chat_response = self.generative_ai_inference_client.chat(self.chat_detail)
            # Extract response text
            response_text = chat_response.data.chat_response.choices[0].message.content[0].text
            print(f"LLM response text is {response_text}")
            logger.info(f"LLM inference successful, response length: {len(response_text)}")
            return response_text
        except RequestException as e:
            if "read timed out" in str(e).lower():
                logger.error("LLM request timed out.")
                raise LLMInferenceError("LLM_TIMEOUT") from e
            logger.error("LLM network error.")
            raise LLMInferenceError("LLM_NETWORK_ERROR") from e
        except ServiceError as e:
            logger.error(f"LLM service error: {e.status}")
            if e.status == 400:
                raise LLMInferenceError("LLM_BAD_REQUEST") from e
            raise LLMInferenceError("LLM_SERVICE_ERROR") from e
        except Exception as e:
            logger.error(f"LLM inference failed: {str(e)}")
            traceback.print_exc()
            raise LLMInferenceError(f"Failed to generate LLM response") from e

    def inference_single_input(self, user_input: str, system_prompt: str) -> str:
        """
        Direct inference for single user input with system prompt - optimized for guard rails.
            This method bypasses chat history and directly processes a single input with a system prompt.
            Args:
                user_input (str): Single user input to validate/process
                system_prompt (str): System prompt for context (e.g., guard rail instructions)
            Returns:
                str: Generated response from the LLM
            Raises:
                Exception: If inference fails
        """
        try:
            # Create OCI messages directly - no chat history conversion needed
            system_message = self._convert_message_to_oci_format("SYSTEM", system_prompt)
            user_message = self._convert_message_to_oci_format("USER", user_input)
            oci_messages = [system_message, user_message]
            # Create chat request with minimal tokens for fast guard rail checks
            chat_request = GenericChatRequest(
                api_format=GenericChatRequest.API_FORMAT_GENERIC,
                messages=oci_messages,
                max_tokens=5000,
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=0.75
            )
            # Set the request and make the call
            self.chat_detail.chat_request = chat_request
            chat_response = self.generative_ai_inference_client.chat(self.chat_detail)
            # Extract response text
            response_text = chat_response.data.chat_response.choices[0].message.content[0].text
            logger.info(f"Single input inference called successfully.")
            return response_text.strip()
        except RequestException as e:
            if "read timed out" in str(e).lower():
                logger.error("LLM request timed out.")
                return "LLM timeout error"
            logger.error("LLM network error.")
            raise LLMInferenceError("LLM_NETWORK_ERROR") from e
        except ServiceError as e:
            logger.error(f"LLM service error: {e.status}")
            if e.status == 400:
                return "It is found that Oracle Genai has marked this in appropriate content and rejected it."
            raise LLMInferenceError("LLM_SERVICE_ERROR") from e
        except Exception as e:
            logger.error(f"Single input inference failed: {str(e)}")
            traceback.print_exc()
            raise LLMInferenceError(f"Failed to generate LLM response") from e


# Convenience function for easy integration with main6.py
def create_llm_client(config_file: str = 'config.ini') -> LLMInference:
    """
    Factory function to create and return an LLM client (default model for text2sql, chat, etc.).
    Args:
        config_file (str): Path to OCI configuration file
    Returns:
        LLMInference: Initialized LLM client
    """
    return LLMInference(config_file)


def create_guardrail_llm_client(config_file: str = 'config.ini') -> LLMInference:
    """
    Factory function to create an LLM client for guardrail checks only.
    Uses GENAI.guard_rail_model_id from config so guardrail can use a different model
    than text2sql/chat.
    Args:
        config_file (str): Path to OCI configuration file
    Returns:
        LLMInference: Initialized LLM client with guardrail model ID
    """
    genai_config = configparser.ConfigParser()
    genai_config.read(config_file)
    guard_model_id = genai_config["GENAI"]["guard_rail_model_id"].strip()
    return LLMInference(config_file, model_id=guard_model_id)
# ---------------------------------------------------------------------------------------------------------------------------------
