# components.py

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from google import genai
from google.genai import types
from groq import Groq

# ---------- LLM wrapper ----------

@dataclass
class GeminiConfig:
    model_name: str = "gemini-2.5-flash"  # fast, cheap; change to a slower "pro" model if you like
    temperature: float = 0.4
    max_output_tokens: int | None = 2048  # or None to let the API decide
    # Thinking-related options
    thinking_budget: int | None = 512     # e.g., 512 tokens for internal reasoning
    include_thoughts: bool = True        # keep False for teaching (no chain-of-thought in output)


@dataclass
class GroqConfig:
    model_name: str = "llama-3.3-70b-versatile"  # or another Groq model
    temperature: float = 0.4
    max_output_tokens: int = 512  # Groq uses max_completion_tokens
    # Reasoning options
    reasoning_format: str | None = "parsed"   # "parsed", "raw", or "hidden", or None
    reasoning_effort: str | None = None       # e.g. "low"/"medium"/"high" for GPT-OSS models
    include_reasoning: bool | None = None     # can't be used together with reasoning_format


class GeminiLLM:
    """
    Wrapper around Google's Gemini API.
    """

    def __init__(self, config: Optional[GeminiConfig] = None, api_key: Optional[str] = None):
        self.config = config or GeminiConfig()

        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "No API key found. Set GEMINI_API_KEY env var or pass api_key to LLM()."
            )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, history=None, return_reasoning = True) -> str:
        contents: list[types.Content] = []

        # Add history if provided
        if history:
            for role, text in history:
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=text)],
                    )
                )

        # Current user message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        # --- NEW: build ThinkingConfig if requested ---
        thinking_cfg = None
        if self.config.thinking_budget is not None:
            thinking_cfg = types.ThinkingConfig(
                thinking_budget=self.config.thinking_budget,
                include_thoughts=self.config.include_thoughts,
            )

        # Build the generation config, including thinking_config
        cfg_kwargs: dict = {
            "temperature": self.config.temperature,
            "thinking_config": thinking_cfg,
        }
        if self.config.max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self.config.max_output_tokens

        gen_config = types.GenerateContentConfig(**cfg_kwargs)

        # Call the model
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=gen_config,
        )

        # Normal path: just use response.text
        # if response.text is not None:
        #     return response.text

        # Fallback: manually assemble visible text from candidates/parts
        pieces: list[str] = []
        answer, thought = "", ""
        for cand in response.candidates or []:
            if not cand.content:
                continue
            for part in cand.content.parts:
                txt = getattr(part, "text", None)
                isThought = getattr(part, "thought", None)
                if isThought:
                    thought = txt
                else:
                    answer = txt
                if txt:
                    pieces.append(txt)

        # return "\n".join(pieces).strip()
        return (answer, thought) if return_reasoning else (answer, "")


class GroqLLM:
    def __init__(self, config: GroqConfig | None = None, api_key: str | None = None):
        self.config = config or GroqConfig()
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY or pass api_key to GroqLLM.")
        self.client = Groq(api_key=api_key)

    def _convert_history_to_messages(self, history):
        messages = []
        if not history:
            return messages
        for role, text in history:
            groq_role = "assistant" if role == "model" else "user" if role == "user" else role
            messages.append({"role": groq_role, "content": text})
        return messages

    def generate(self, prompt: str, history=None, return_reasoning: bool = True):
        messages = self._convert_history_to_messages(history)
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
        )

        # Reasoning options
        if self.config.reasoning_format is not None:
            kwargs["reasoning_format"] = self.config.reasoning_format
        elif self.config.include_reasoning is not None:
            kwargs["include_reasoning"] = self.config.include_reasoning

        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        chat_completion = self.client.chat.completions.create(**kwargs)
        msg = chat_completion.choices[0].message

        answer = msg.content
        reasoning = getattr(msg, "reasoning", None)

        if not return_reasoning:
            return answer, ""

        # For raw mode: split out <think>...</think> if needed
        if self.config.reasoning_format == "raw" and reasoning is None:
            import re
            think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        return answer, reasoning

# ---------- Prompt generator ----------

class PromptGenerator:
    """
    Generates task-specific prompts for our salary-hike scenario.

    We keep base_context for reusability and to show 'system-style'
    instructions inside the prompt.
    """

    def __init__(self, base_context: Optional[str] = None):
        self.base_context = base_context or (
            "You are an experienced HR and organizational-behavior expert. "
            "You follow standard best practices for performance reviews and "
            "fair salary decisions."
        )

    def _with_context(self, task_prompt: str) -> str:
        return f"{self.base_context}\n\n{task_prompt}"

    @staticmethod
    def format_employee_descriptions(employees: Dict[str, str]) -> str:
        """
        employees: mapping like {'A': 'Alice, high performer...', 'B': 'Bob, ...'}
        """
        lines = []
        for label, desc in employees.items():
            lines.append(f"{label}. {desc}")
        return "\n".join(lines)

    # --- (i) MCQ-style prompt ---

    def mcq_salary_hike(self, employees: Dict[str, str], 
                        deserve = "deserve", hike = "hike") -> str:
        """
        Question 1: 'Which among A, B, C, and D deserve a salary hike? (MCQ)'
        We instruct the model to answer in a strict format:
            <answer>A,B,D</answer>
        """

        employee_text = self.format_employee_descriptions(employees)

        task = f"""
You are evaluating employees for a potential salary hike.

Here are the employees:
{employee_text}

Question:
Which among A, B, C, and D {deserve} a salary {hike}?

Requirements:
- Consider job performance, impact, consistency, and fairness.
- You may select multiple employees, or none at all.
- Respond in EXACTLY one line, using this format:

<answer>comma-separated list of options from {{A,B,C,D}}</answer>

Examples:
- <answer>A,B</answer>
- <answer>C</answer>
- <answer>(none)</answer>  # if you believe nobody deserves a raise

Do NOT add any explanation or additional text.
"""
        return self._with_context(task.strip())

    # --- (ii) Free-flow / unconstrained answer ---

    def free_flow_criteria(self) -> str:
        """
        Question 2: 'What are the criteria a manager should use to decide about salary hikes?'
        No strict structure, just good content.
        """

        task = """
Managers often need to decide which employees receive salary hikes.

Question:
What are the key criteria a manager should use to decide about salary hikes?

Please provide a concise answer (a short paragraph or a bullet list).
"""
        return self._with_context(task.strip())

    # --- (iii) Structured JSON output ---

    def structured_json_raises(self, employees: Dict[str, str]) -> str:
        """
        Question 3: 'List all the employees who should receive a raise and by how much' (JSON).
        We ask for a strict JSON object that we can parse programmatically.
        """

        employee_text = self.format_employee_descriptions(employees)

        task = f"""
You must decide which employees receive a salary raise.

Here are the employees:
{employee_text}

Your task:
Return ONLY a valid JSON object that follows this exact schema:

{{
  "employees": [
    {{
      "id": "A",                    // one of A, B, C, D
      "recommended_raise_percent": 5.0,
      "justification": "Short explanation of why this employee deserves this raise."
    }},
    ...
  ]
}}

Guidelines:
- Include only employees who should receive a raise.
- "recommended_raise_percent" is a float between 0 and 20.
- "justification" should be 1-2 sentences.
- Do NOT include comments in the JSON.
- Do NOT wrap the JSON in backticks or additional text; output JSON ONLY.
"""
        return self._with_context(task.strip())
