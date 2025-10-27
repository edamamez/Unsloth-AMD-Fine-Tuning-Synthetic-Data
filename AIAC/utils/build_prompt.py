#!/usr/bin/python3
import json
from typing import List


def option_extractor_prompt(answer_string: str, choices: List[str]) -> str:
    tmpl = """
        You are an advanced LLM specialized in extracting the single correct option letter (A, B, C, D) from answers to multiple-choice questions. Your job is to parse the provided answer text, remove unnecessary prefixes, and identify which letter best matches the answer. If no direct letter is apparent, use textual comparison with the provided choices. If still no match is found, return "X".

        EXTRACTION RULES:
        1. Direct Letter Matching (e.g., "A", "(B)", or "The correct answer is C").
        2. Prefix Removal (e.g., "The answer is", "Best option:", etc.).
        3. Text Matching: If no letter is found, compare the answer text with the option contents.

        OUTPUT FORMAT: Return only the correct option letter (A, B, C, or D), with no additional text.

        Example 1:
        Answer String: "The correct answer is B"
        Choices: ["A) 21", "B) 44", "C) 15", "D) 68"]
        Output: B

        Example 2:
        Answer String: "Photosynthesis happens in chloroplasts"
        Choices: ["A) Chloroplasts are responsible for photosynthesis", "B) Photosynthesis occurs in mitochondria", "C) The photosynthetic process is responsible for oxygen release", "D) Light absorption enables this process"]
        Output: A

        Answer String: {}
        Choices: {}
        Output:
        """
    return tmpl.format(answer_string, json.dumps(choices))


def auto_json(json_str: str) -> str:
    prompt = """
        You are a JSON fixer. Given a malformed or incomplete JSON-like string, your task is to fix any missing or incorrect syntax such as braces, brackets, commas, colons, double quotes, or other JSON formatting issues, so that the result is valid JSON that can be parsed using `json.loads()` in Python.

        RULES:
        1. All object keys and string values must be enclosed in double quotes (").
        2. Boolean values must be `true` or `false` (lowercase), and null must be `null`.
        3. Remove any trailing commas before closing braces or brackets.
        4. Do not change any actual values (like numbers or strings) unless necessary to fix JSON syntax.
        5. **Remove JSON code block text with backticks** like **```json** and **```**.

        Return only the corrected JSON string.
        Example Input:
        ```json{{"name": "Alice", "age": 30, "skills": ["Python", "ML"],}}```

        Example Output:
        {{
            "name": "Alice",
            "age": 30,
            "skills": ["Python", "ML"]
        }}

        Example Input:
        The answer is: {{"name": "Bob", "age": 25, "skills": ["Java", "AI"],}}

        Example Output:
        {{
            "name": "Bob",
            "age": 25,
            "skills": ["Java", "AI"]
        }}

        Input:
        {}
        """

    return prompt.format(json_str)
