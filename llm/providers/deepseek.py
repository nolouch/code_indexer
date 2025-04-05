import os
import logging
import json
from typing import Optional, Generator
import requests

from llm.base import BaseLLMProvider
# Import DeepSeek API configuration from the new fileindexer_llm file
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from setting.fileindexer_llm import FILEINDER_DEEPSEEK_API_KEY, FILEINDER_DEEPSEEK_API_BASE

logger = logging.getLogger(__name__)

class DeepSeekProvider(BaseLLMProvider):
    """
    Provider for the DeepSeek API
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs):
        """
        Initialize the DeepSeek provider
        
        Args:
            model: The model to use
            api_key: DeepSeek API key (defaults to FILEINDER_DEEPSEEK_API_KEY env var)
            api_base: DeepSeek API base URL (defaults to FILEINDER_DEEPSEEK_API_BASE env var or standard endpoint)
        """
        super().__init__(model, **kwargs)
        # First try to use passed api_key, then use FILEINDER_DEEPSEEK_API_KEY from settings
        self.api_key = api_key or FILEINDER_DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Please set FILEINDER_DEEPSEEK_API_KEY environment variable.")
            
        # Use API base URL from settings
        self.api_base = api_base or FILEINDER_DEEPSEEK_API_BASE
        logger.info(f"Initialized DeepSeek provider with model: {model}")

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        """Build the messages array for the API request"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Generate text using the DeepSeek API
        
        Args:
            prompt: The prompt to send to the API
            system_prompt: Optional system prompt
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            The generated text
        """
        messages = self._build_messages(prompt, system_prompt)
        
        # Update kwargs with default configuration
        kwargs = self._update_kwargs(kwargs)
        
        def _generate():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
                
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        return self._retry_with_exponential_backoff(_generate)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """
        Generate a streaming response from the DeepSeek API
        
        Args:
            prompt: The prompt to send to the API
            system_prompt: Optional system prompt
            **kwargs: Additional parameters to pass to the API
            
        Yields:
            Chunks of the generated text
        """
        messages = self._build_messages(prompt, system_prompt)
        
        # Update kwargs with default configuration
        kwargs = self._update_kwargs(kwargs)
        kwargs["stream"] = True
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"] 