"""LLM API client for handling interactions with different language model providers."""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import anthropic
import openai
import boto3
import groq
import writerai
import google.generativeai as genai
from xai import Grok

class LLMClient:
    """Client for interacting with various LLM APIs."""

    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the LLM client with API keys."""
        self.api_keys = api_keys
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize API clients for different providers."""
        try:
            # Anthropic
            if self.api_keys.get("ANTHROPIC_API_KEY"):
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_keys["ANTHROPIC_API_KEY"])
            else:
                self.anthropic_client = None

            # OpenAI
            if self.api_keys.get("OPENAI_API_KEY"):
                self.openai_client = openai.OpenAI(api_key=self.api_keys["OPENAI_API_KEY"])
            else:
                self.openai_client = None

            # AWS Bedrock
            if self.api_keys.get("AWS_ACCESS_KEY_ID") and self.api_keys.get("AWS_SECRET_ACCESS_KEY"):
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    aws_access_key_id=self.api_keys["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=self.api_keys["AWS_SECRET_ACCESS_KEY"],
                    region_name=self.api_keys.get("AWS_REGION", "us-east-1")
                )
            else:
                self.bedrock_client = None

            # Groq
            if self.api_keys.get("GROQ_API_KEY"):
                self.groq_client = groq.Groq(api_key=self.api_keys["GROQ_API_KEY"])
            else:
                self.groq_client = None

            # Writer
            if self.api_keys.get("WRITER_API_KEY"):
                self.writer_client = writerai.Client(api_key=self.api_keys["WRITER_API_KEY"])
            else:
                self.writer_client = None

            # Gemini
            if self.api_keys.get("GEMINI_API_KEY"):
                genai.configure(api_key=self.api_keys["GEMINI_API_KEY"])
                self.gemini_client = genai
            else:
                self.gemini_client = None

            # XAI (Grok)
            if self.api_keys.get("XAI_API_KEY"):
                self.xai_client = Grok(api_key=self.api_keys["XAI_API_KEY"])
            else:
                self.xai_client = None

        except Exception as e:
            logging.error(f"Error initializing LLM clients: {e}")
            raise

    def _call_anthropic(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call Anthropic's API."""
        try:
            response = self.anthropic_client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **parameters
            )
            return response.content[0].text, {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            }
        except Exception as e:
            logging.error(f"Error calling Anthropic API: {e}")
            raise

    def _call_openai(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call OpenAI's API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **parameters
            )
            return response.choices[0].message.content, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            raise

    def _call_bedrock(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call AWS Bedrock's API."""
        try:
            request_body = {
                "prompt": prompt,
                **parameters
            }
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            return response_body['completion'], {
                "prompt_tokens": response_body.get('usage', {}).get('prompt_tokens', 0),
                "completion_tokens": response_body.get('usage', {}).get('completion_tokens', 0)
            }
        except Exception as e:
            logging.error(f"Error calling Bedrock API: {e}")
            raise

    def _call_groq(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call Groq's API."""
        try:
            response = self.groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **parameters
            )
            return response.choices[0].message.content, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            logging.error(f"Error calling Groq API: {e}")
            raise

    def _call_writer(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call Writer's API."""
        try:
            response = self.writer_client.completions.create(
                model=model_id,
                prompt=prompt,
                **parameters
            )
            return response.text, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            logging.error(f"Error calling Writer API: {e}")
            raise

    def _call_gemini(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call Google's Gemini API."""
        try:
            model = self.gemini_client.GenerativeModel(model_id)
            response = model.generate_content(prompt, **parameters)
            return response.text, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            logging.error(f"Error calling Gemini API: {e}")
            raise

    def _call_xai(self, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call XAI's (Grok) API."""
        try:
            response = self.xai_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **parameters
            )
            return response.choices[0].message.content, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            logging.error(f"Error calling XAI API: {e}")
            raise

    def call_model(self, model_type: str, model_id: str, prompt: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """Call the appropriate model based on type."""
        start_time = time.time()
        try:
            if model_type == "anthropic":
                if not self.anthropic_client:
                    raise ValueError("Anthropic client not initialized")
                response, usage = self._call_anthropic(model_id, prompt, parameters)
            elif model_type == "openai":
                if not self.openai_client:
                    raise ValueError("OpenAI client not initialized")
                response, usage = self._call_openai(model_id, prompt, parameters)
            elif model_type == "bedrock":
                if not self.bedrock_client:
                    raise ValueError("Bedrock client not initialized")
                response, usage = self._call_bedrock(model_id, prompt, parameters)
            elif model_type == "groq":
                if not self.groq_client:
                    raise ValueError("Groq client not initialized")
                response, usage = self._call_groq(model_id, prompt, parameters)
            elif model_type == "writer":
                if not self.writer_client:
                    raise ValueError("Writer client not initialized")
                response, usage = self._call_writer(model_id, prompt, parameters)
            elif model_type == "gemini":
                if not self.gemini_client:
                    raise ValueError("Gemini client not initialized")
                response, usage = self._call_gemini(model_id, prompt, parameters)
            elif model_type == "xai":
                if not self.xai_client:
                    raise ValueError("XAI client not initialized")
                response, usage = self._call_xai(model_id, prompt, parameters)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            elapsed_time = time.time() - start_time
            logging.info(f"Model {model_id} response time: {elapsed_time:.2f}s")
            return response, usage

        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(f"Error calling model {model_id} after {elapsed_time:.2f}s: {e}")
            raise 