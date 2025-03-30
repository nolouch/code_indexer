import re
from typing import Optional, Tuple, Dict


def extract_json(response: str) -> str:
    """Extract JSON from the plan response."""
    json_code_block_pattern = re.compile(
        r"```json\s*(\[\s*{.*?}\s*\])\s*```", re.DOTALL
    )
    match = json_code_block_pattern.search(response)
    if match:
        return match.group(1)

    json_code_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_code_block_pattern.search(response)
    if match:
        return match.group(1)

    json_str = find_first_json_object(response)
    if not json_str:
        raise ValueError("No valid JSON array found in the response.")

    return json_str


def extract_json_array(response: str) -> str:
    """Extract JSON array from the plan response."""
    json_code_block_pattern = re.compile(
        r"```json\s*(\[\s*{.*?}\s*\])\s*```", re.DOTALL
    )
    match = json_code_block_pattern.search(response)
    if match:
        return match.group(1)

    json_code_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_code_block_pattern.search(response)
    if match:
        return match.group(1)

    json_str = find_first_json_array(response)
    if not json_str:
        raise ValueError("No valid JSON array found in the response.")

    return json_str


def find_first_json_object(text: str) -> Optional[str]:
    """Find the first JSON object in the given text."""
    stack = []
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append(i)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
    return None


def find_first_json_array(text: str) -> Optional[str]:
    """Find the first array in the given text."""
    stack = []
    start = -1
    for i, char in enumerate(text):
        if char == "[":
            if not stack:
                start = i
            stack.append(i)
        elif char == "]":
            if stack:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
    return None
