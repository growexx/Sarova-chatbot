import pytest
from unittest.mock import MagicMock, patch

from code_modules.oracle_genai_handler import (
    LLMInference,
    create_llm_client,
    LLMInferenceError
)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_chat_response():
    """Mock OCI chat response structure"""
    mock_content = MagicMock()
    mock_content.text = "Mock LLM response"

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_chat_response = MagicMock()
    mock_chat_response.data.chat_response.choices = [mock_choice]

    return mock_chat_response


@pytest.fixture
def mock_oci_client(mock_chat_response):
    """Mock Generative AI inference client"""
    client = MagicMock()
    client.chat.return_value = mock_chat_response
    return client


# -----------------------------
# Initialization Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_llm_inference_initialization(
    mock_client_class,
    mock_from_file,
):
    """Test LLMInference initializes successfully"""
    mock_from_file.return_value = {"config": "ok"}
    mock_client_class.return_value = MagicMock()

    client = LLMInference("config.ini")

    assert client.generative_ai_inference_client is not None
    assert client.chat_detail is not None


@patch("code_modules.oracle_genai_handler.oci.config.from_file", side_effect=Exception("Config error"))
def test_llm_inference_initialization_failure(mock_from_file):
    """Test initialization failure raises exception"""
    with pytest.raises(Exception):
        LLMInference("config.ini")


# -----------------------------
# Message Conversion Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_convert_message_to_oci_format(mock_client, mock_from_file):
    """Test role + content conversion"""
    mock_from_file.return_value = {}
    mock_client.return_value = MagicMock()

    client = LLMInference()

    msg = client._convert_message_to_oci_format("USER", "Hello")

    assert msg.role == "USER"
    assert msg.content[0].text == "Hello"


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_convert_chat_history_to_oci_format(mock_client, mock_from_file):
    """Test chat history conversion and filtering"""
    mock_from_file.return_value = {}
    mock_client.return_value = MagicMock()

    client = LLMInference()

    chat_history = [
        {"role": "USER", "message": "Hi", "timestamp": ""},
        {"role": "ASSISTANT", "message": "", "timestamp": ""},  # should be skipped
    ]

    messages = client._convert_chat_history_to_oci_format(chat_history)

    assert len(messages) == 1
    assert messages[0].content[0].text == "Hi"


# -----------------------------
# Inference Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_from_chat_history_success(
    mock_client_class,
    mock_from_file,
    mock_oci_client,
):
    """Test successful chat-history inference"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = mock_oci_client

    client = LLMInference()

    response = client.inference_from_chat_history(
        [{"role": "USER", "message": "Hello", "timestamp": ""}]
    )

    assert response[0] == "M"
    mock_oci_client.chat.assert_called_once()


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_from_empty_chat_history(
    mock_client_class,
    mock_from_file,
):
    """Test empty chat history returns safe message"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = MagicMock()

    client = LLMInference()

    response = client.inference_from_chat_history(
        [{"role": "USER", "message": "   ", "timestamp": ""}]
    )

    assert "didn't receive any valid messages" in response


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_from_chat_history_failure(
    mock_client_class,
    mock_from_file,
):
    """Test inference failure propagates exception"""
    mock_from_file.return_value = {}
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("OCI failure")
    mock_client_class.return_value = mock_client

    client = LLMInference()

    with pytest.raises(Exception):
        client.inference_from_chat_history(
            [{"role": "USER", "message": "Hello", "timestamp": ""}]
        )

import pytest
from unittest.mock import MagicMock, patch

from code_modules.oracle_genai_handler import LLMInference


@pytest.fixture
def llm():
    return LLMInference()


def test_convert_chat_history_to_oci_format_exception(llm):
    # Arrange
    chat_history = [
        {"role": "USER", "message": "Hello"},
    ]

    # Force internal method to raise exception
    llm._convert_message_to_oci_format = MagicMock(
        side_effect=Exception("boom")
    )

    # Act + Assert
    with pytest.raises(Exception) as exc:
        llm._convert_chat_history_to_oci_format(chat_history)

    assert "boom" in str(exc.value)

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_single_input(
    mock_client_class,
    mock_from_file,
    mock_oci_client,
):
    """Test single-input inference"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = mock_oci_client

    client = LLMInference()

    response = client.inference_single_input("Hello", "System prompt")

    assert response[0] == "M"


# -----------------------------
# Factory Function Test
# -----------------------------

@patch("code_modules.oracle_genai_handler.LLMInference")
def test_create_llm_client(mock_llm):
    """Test factory function"""
    create_llm_client("config.ini")
    mock_llm.assert_called_once_with("config.ini")



import pytest
from unittest.mock import MagicMock, patch

from code_modules.oracle_genai_handler import LLMInference


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_single_input_exception(
    mock_client_class,
    mock_from_file,
):
    mock_from_file.return_value = {}

    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("OCI failure")
    mock_client_class.return_value = mock_client

    client = LLMInference()

    with pytest.raises(LLMInferenceError) as exc:
        client.inference_single_input(
            user_input="test input",
            system_prompt="system prompt"
        )

    assert "Failed to generate LLM response" in str(exc.value)
    assert exc.value.__cause__ is not None
    assert "OCI failure" in str(exc.value.__cause__)


import pytest
from unittest.mock import MagicMock
from requests.exceptions import RequestException
from oci.exceptions import ServiceError

from code_modules.oracle_genai_handler import (
    LLMInference,
    LLMInferenceError
)


# ---------------------------------------------------------
# Helper to create mocked LLMInference instance
# ---------------------------------------------------------

@pytest.fixture
def mock_llm():
    llm = LLMInference.__new__(LLMInference)

    llm.generative_ai_inference_client = MagicMock()
    llm.chat_detail = MagicMock()

    llm._convert_chat_history_to_oci_format = MagicMock(
        return_value=[{"role": "USER", "content": "hi"}]
    )

    llm._convert_message_to_oci_format = MagicMock(
        side_effect=lambda role, msg: {"role": role, "content": msg}
    )

    return llm


# =========================================================
# 🔥 inference_from_chat_history EXCEPTION TESTS
# =========================================================

def test_chat_history_timeout(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        RequestException("Read timed out")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history([{"role": "USER", "message": "hi"}])

    assert str(exc.value) == "Failed to generate LLM response"


def test_chat_history_network_error(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        RequestException("Connection aborted")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history([{"role": "USER", "message": "hi"}])

    assert str(exc.value) == "Failed to generate LLM response"


def test_chat_history_service_error_400(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        ServiceError(status=400, code="400",headers={},message="Bad request")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history([{"role": "USER", "message": "hi"}])

    assert str(exc.value) == "LLM_BAD_REQUEST"


def test_chat_history_service_error_other(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        ServiceError(status=500, code="500", headers={},message="Internal error")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history([{"role": "USER", "message": "hi"}])

    assert str(exc.value) == "LLM_SERVICE_ERROR"


def test_chat_history_generic_exception(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        Exception("Unexpected failure")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history([{"role": "USER", "message": "hi"}])

    assert "Failed to generate LLM response" in str(exc.value)


# =========================================================
# 🔥 inference_single_input EXCEPTION TESTS
# =========================================================

# def test_single_input_timeout_returns_message(mock_llm):
#     mock_llm.generative_ai_inference_client.chat.side_effect = \
#         RequestException("Read timed out")

#     with pytest.raises(LLMInferenceError) as exc:
#         mock_llm.inference_single_input("hi", "system")

#     assert str(exc.value) == "Failed to generate LLM response"

def test_single_input_network_error(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        RequestException("Connection error")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_single_input("hi", "system")

    assert str(exc.value) == "Failed to generate LLM response"


def test_single_input_service_error_400_returns_message(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        ServiceError(status=400, code="400",headers={}, message="Bad request")

    result = mock_llm.inference_single_input("hi", "system")

    assert "Oracle Genai has marked this" in result


def test_single_input_service_error_other(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        ServiceError(status=500, code="500",headers={}, message="Internal error")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_single_input("hi", "system")

    assert str(exc.value) == "LLM_SERVICE_ERROR"


def test_single_input_generic_exception(mock_llm):
    mock_llm.generative_ai_inference_client.chat.side_effect = \
        Exception("Unexpected failure")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_single_input("hi", "system")

    assert "Failed to generate LLM response" in str(exc.value)
    
def test_chat_history_timeout(mock_llm):
    from code_modules.oracle_genai_handler import RequestException

    mock_llm.generative_ai_inference_client.chat.side_effect = \
        RequestException("Read timed out")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history(
            [{"role": "USER", "message": "hi"}]
        )

    assert str(exc.value) == "LLM_TIMEOUT"

def test_chat_history_network_error(mock_llm):
    from code_modules.oracle_genai_handler import RequestException

    mock_llm.generative_ai_inference_client.chat.side_effect = \
        RequestException("Connection aborted")

    with pytest.raises(LLMInferenceError) as exc:
        mock_llm.inference_from_chat_history(
            [{"role": "USER", "message": "hi"}]
        )

    assert str(exc.value) == "LLM_NETWORK_ERROR"

from oci.exceptions import RequestException, ServiceError

# Correct test for timeout branch
def test_single_input_timeout():
    mock_llm = LLMInference()
    mock_llm.generative_ai_inference_client = MagicMock()
    mock_llm.chat_detail = MagicMock()

    # Message must contain "read timed out" (case-insensitive)
    mock_llm.generative_ai_inference_client.chat.side_effect = RequestException("Read timed out")  # OK, .lower() will catch

    result = mock_llm.inference_single_input("Hello", "System prompt")
    assert result == "LLM timeout error"


# Correct test for network error branch
def test_single_input_network_error():
    mock_llm = LLMInference()
    mock_llm.generative_ai_inference_client = MagicMock()
    mock_llm.chat_detail = MagicMock()

    # Any other RequestException triggers LLM_NETWORK_ERROR
    mock_llm.generative_ai_inference_client.chat.side_effect = RequestException("Connection aborted")

    with pytest.raises(LLMInferenceError) as exc_info:
        mock_llm.inference_single_input("Hello", "System prompt")

    # The exception raised in the code is exactly "LLM_NETWORK_ERROR"
    assert str(exc_info.value) == "LLM_NETWORK_ERROR"