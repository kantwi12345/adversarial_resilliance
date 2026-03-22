"""Universal LLM Adapter with Adversarial Resilience.

This module provides a base class for integrating ANY LLM with
the adversarial resilience framework. Just implement _generate().
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from adaptive_architect import (
    TaskRequest,
    RuntimeTier,
    StructuralFilter,
    ExecutionGuard,
    IntentAnalyzer,
    build_default_tdg,
    TraceSegment,
)


# ═══════════════════════════════════════════════════════════════
# EMBEDDING-BASED INTENT ANALYZER
# ═══════════════════════════════════════════════════════════════

class EmbeddingIntentAnalyzer(IntentAnalyzer):
    """Uses sentence-transformers for semantic similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.55,
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def is_intent_consistent(self, goal: str, incoming_text: str) -> bool:
        # First check regex patterns (fast rejection)
        for pat in self._SUSPICIOUS_PATTERNS:
            if pat.search(incoming_text):
                return False

        # Then check semantic similarity
        embeddings = self.model.encode([goal, incoming_text])
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(embeddings[0], embeddings[1]) / (
            norm(embeddings[0]) * norm(embeddings[1])
        )
        return similarity >= self.threshold


# ═══════════════════════════════════════════════════════════════
# BASE LLM RUNTIME (ABSTRACT)
# ═══════════════════════════════════════════════════════════════

class BaseLLMRuntime(ABC):
    """Abstract base class for any LLM integration.

    Subclasses only need to implement:
        - _generate(system_prompt, user_message) -> str
        - get_model_name() -> str

    The safety pipeline is handled automatically.
    """

    def __init__(self, use_embedding_analyzer: bool = True):
        tdg = build_default_tdg()
        self.runtime = RuntimeTier(tdg)

        # Use embedding-based analyzer for better accuracy
        if use_embedding_analyzer:
            self.runtime.intent_analyzer = EmbeddingIntentAnalyzer()

    def execute(self, request: TaskRequest) -> str:
        """Run safety checks, then generate response."""

        # Clear previous armor state
        self.runtime.agent_armor.clear()

        # Step 1: Intent check + structural filter + execution guard
        if not self.runtime.execute_task(request):
            raise RuntimeError("Request blocked by safety checks")

        # Step 2: Log trace for Agent Armor
        for idx, tool in enumerate(request.planned_tools, start=1):
            self.runtime.agent_armor.log_segment(
                TraceSegment(
                    tool=tool,
                    source="trusted" if idx == 1 else "sanitized",
                    data=request.raw_input,
                    sink="high_privilege" if tool == "write" else None,
                )
            )

        # Step 3: Validate data flow
        if not self.runtime.agent_armor.validate():
            raise RuntimeError("Agent Armor blocked unsafe data flow")

        # Step 4: Build system prompt
        system_prompt = request.goal
        if request.system_instructions:
            system_prompt = f"{request.system_instructions}\n\nTask: {request.goal}"

        # Step 5: Generate (implemented by subclass)
        return self._generate(system_prompt, request.raw_input)

    @abstractmethod
    def _generate(self, system_prompt: str, user_message: str) -> str:
        """Generate response from the LLM. Override in subclass."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return display name of the model."""
        pass


# ═══════════════════════════════════════════════════════════════
# TINYLLAMA IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════

class TinyLlamaRuntime(BaseLLMRuntime):
    """TinyLlama with adversarial resilience."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__(use_embedding_analyzer=True)

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float32,
            device="cpu",
        )
        self._model_name = model_name

    def _generate(self, system_prompt: str, user_message: str) -> str:
        formatted = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"

        response = self.pipe(
            formatted,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        text = response[0]["generated_text"]

        # Extract only the assistant's response
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1].strip()
        if "</s>" in text:
            text = text.split("</s>")[0].strip()

        return text

    def get_model_name(self) -> str:
        return "TinyLlama-1.1B-Chat"


# ═══════════════════════════════════════════════════════════════
# OPENAI IMPLEMENTATION (Optional - needs API key)
# ═══════════════════════════════════════════════════════════════

class OpenAIRuntime(BaseLLMRuntime):
    """OpenAI GPT with adversarial resilience."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__(use_embedding_analyzer=True)

        import openai
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

        self.model = model

    def _generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return f"OpenAI {self.model}"


# ═══════════════════════════════════════════════════════════════
# OLLAMA IMPLEMENTATION (Local models)
# ═══════════════════════════════════════════════════════════════

class OllamaRuntime(BaseLLMRuntime):
    """Ollama local models with adversarial resilience."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        super().__init__(use_embedding_analyzer=True)
        self.model = model
        self.base_url = base_url

    def _generate(self, system_prompt: str, user_message: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"System: {system_prompt}\n\nUser: {user_message}",
                "stream": False,
            },
            timeout=120,
        )
        return response.json().get("response", "")

    def get_model_name(self) -> str:
        return f"Ollama {self.model}"


# ═══════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════

def create_runtime(model_choice: str, **kwargs) -> BaseLLMRuntime:
    """Factory function to create the appropriate runtime."""

    if model_choice == "TinyLlama":
        return TinyLlamaRuntime()
    elif model_choice == "GPT-4o Mini":
        return OpenAIRuntime(model="gpt-4o-mini", **kwargs)
    elif model_choice == "GPT-4":
        return OpenAIRuntime(model="gpt-4", **kwargs)
    elif model_choice == "Ollama Llama3":
        return OllamaRuntime(model="llama3", **kwargs)
    elif model_choice == "Ollama Mistral":
        return OllamaRuntime(model="mistral", **kwargs)
    else:
        return TinyLlamaRuntime()
