"""LLM provider and model configuration settings."""
import os

# LLM Provider Settings
FILEINDER_LLM_PROVIDER = os.environ.get("FILEINDER_LLM_PROVIDER", "deepseek")
FILEINDER_LLM_MODEL = os.environ.get("FILEINDER_LLM_MODEL", "deepseek-coder-v2-instruct")
FILEINDER_DEEPSEEK_API_KEY = os.environ.get("FILEINDER_DEEPSEEK_API_KEY")
FILEINDER_OPENAI_API_KEY = os.environ.get("FILEINDER_OPENAI_API_KEY")
FILEINDER_OLLAMA_BASE_URL = os.environ.get("FILEINDER_OLLAMA_BASE_URL", "http://localhost:11434")

# DeepSeek API default settings
FILEINDER_DEEPSEEK_API_BASE = os.environ.get("FILEINDER_DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# LLM comment generation settings
FILEINDER_GENERATE_COMMENTS = os.environ.get("FILEINDER_GENERATE_COMMENTS", "true").lower() == "true"
FILEINDER_COMMENTS_MAX_LENGTH = int(os.environ.get("FILEINDER_COMMENTS_MAX_LENGTH", "64000"))

# LLM comment generation system prompt
FILEINDER_COMMENTS_SYSTEM_PROMPT = os.environ.get(
    "FILEINDER_COMMENTS_SYSTEM_PROMPT", 
    """You are an expert programmer who specializes in code understanding.
Describe the code in less than 5~10 lines. Focus on the main purpose and functionality.
Do not add fluff like "the code you provided" or "this code". Be concise and technical."""
)
