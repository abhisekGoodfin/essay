import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the application.
    Encapsulates all configurable parameters for easy modification.
    """
    RESULTS_DIR = "./results"
    CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")

    SAVE_INTERMEDIATE_RESULTS = True
    SAVE_INTERVAL = 10
    REQUEST_EXPLANATIONS = True

    JSON_OUTPUT_FILE = "updated_data_essay.json"
    COMPARISON_SUMMARY_FILE = os.path.join(RESULTS_DIR, "model_comparison_summary.json")

    # API Keys - loaded from environment
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    WRITER_API_KEY = os.getenv("WRITER_API_KEY", "")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    # Model selection flags
    RUN_ANTHROPIC_MODELS = True
    RUN_BEDROCK_MODELS = True
    RUN_OPENAI_MODELS = True
    RUN_GEMINI_MODELS = True
    RUN_GROQ_MODELS = True
    RUN_WRITER_MODELS = True
    RUN_XAI_MODELS = True

    # Selected models to run (empty list means use flags above)
    SELECTED_MODELS: List[str] = []

    # API costs per million tokens
    API_COSTS = {
        "anthropic": {
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
            "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00}
        },
        "openai": {
            "gpt-4o": {"input": 5.00, "output": 15.00},
            "gpt-4.1-2025-04-14": {"input": 10.00, "output": 30.00},
            "gpt-4.1-mini-2025-04-14": {"input": 1.00, "output": 3.00},
            "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.30}
        },
        "bedrock": {
            "anthropic.claude-3-opus-20240229-v1:0": {"input": 15.00, "output": 75.00},
            "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 3.00, "output": 15.00},
            "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.25, "output": 1.25},
            "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 3.00, "output": 15.00},
            "anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 3.00, "output": 15.00},
            "mistral.mistral-large-2402-v1:0": {"input": 8.00, "output": 24.00},
        },
        "groq": {
            "deepseek-r1-distill-llama-70b": {"input": 0.59, "output": 0.79},
            "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.27, "output": 0.38},
            "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.27, "output": 0.38}
        },
        "gemini": {
            "models/gemini-2.5-pro-preview-03-25": {"input": 0.35, "output": 1.05},
            "models/gemini-1.5-flash-latest": {"input": 0.35, "output": 0.53},
            "models/gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
        },
        "writer": {
            "palmyra-fin": {"input": 1.00, "output": 3.00}
        },
        "xai": {
            "grok-3-latest": {"input": 5.00, "output": 15.00}
        }
    }

    # Model configurations
    MODEL_CONFIGS = [
        {
            "config_id": "claude-3.7-sonnet",
            "type": "anthropic",
            "model_id": "claude-3-7-sonnet-20250219",
            "parameters": {
                "temperature": 0.1, "max_tokens": 1024,
                "top_p": 0.999, "top_k": 250
            }
        },
        {
            "config_id": "claude-3.5-sonnet",
            "type": "anthropic",
            "model_id": "claude-3-5-sonnet-20240620",
            "parameters": {
                "temperature": 0.1, "max_tokens": 1024,
                "top_p": 0.999, "top_k": 250
            }
        },
        {
            "config_id": "claude-3-haiku",
            "type": "anthropic",
            "model_id": "claude-3-haiku-20240307",
            "parameters": {
                "temperature": 0.1, "max_tokens": 1024,
                "top_p": 0.999, "top_k": 250
            }
        },
        {
            "config_id": "mistral-large",
            "type": "bedrock",
            "model_id": "mistral.mistral-large-2402-v1:0",
            "parameters": {
                "temperature": 0.1, "top_p": 0.9, "top_k": 50, "max_tokens": 1024
            }
        },
        {
            "config_id": "gpt-4o", 
            "type": "openai", 
            "model_id": "gpt-4o",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "gpt-4.1", 
            "type": "openai", 
            "model_id": "gpt-4.1-2025-04-14",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "gpt-4.1-mini", 
            "type": "openai", 
            "model_id": "gpt-4.1-mini-2025-04-14",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "gpt-4.1-nano", 
            "type": "openai", 
            "model_id": "gpt-4.1-nano-2025-04-14",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "grok-3", 
            "type": "xai", 
            "model_id": "grok-3-latest",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024
            }
        },
        {
            "config_id": "gemini-2.5-pro", 
            "type": "gemini", 
            "model_id": "models/gemini-2.5-pro-preview-03-25",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "top_k": 40, 
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            }
        },
        {
            "config_id": "gemini-2.5-flash", 
            "type": "gemini", 
            "model_id": "models/gemini-2.0-flash",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "top_k": 40, 
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            }
        },
        {
            "config_id": "gemini-1.5-pro", 
            "type": "gemini", 
            "model_id": "models/gemini-1.5-pro-latest",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "top_k": 40, 
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            }
        },
        {
            "config_id": "gemini-1.5-flash", 
            "type": "gemini", 
            "model_id": "models/gemini-1.5-flash-latest",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "top_k": 40, 
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            }
        },
        {
            "config_id": "palmyra-fin", 
            "type": "writer", 
            "model_id": "palmyra-fin",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024
            }
        },
        {
            "config_id": "llama-3-70b", 
            "type": "groq", 
            "model_id": "llama3-70b-8192",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "deepseek-r1-distill-llama-70b", 
            "type": "groq", 
            "model_id": "deepseek-r1-distill-llama-70b",
            "parameters": {
                "temperature": 0.1, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "llama-4-maverick-17b", 
            "type": "groq", 
            "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        },
        {
            "config_id": "llama-4-scout-17b", 
            "type": "groq", 
            "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
            "parameters": {
                "temperature": 0.1, 
                "top_p": 0.95, 
                "max_tokens": 1024, 
                "response_format": {"type": "json_object"}
            }
        }
    ]