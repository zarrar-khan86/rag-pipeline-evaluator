"""
Answer generation from retrieved context.
Supports provider-backed LLMs when configured, with a deterministic fallback.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


class AnswerGenerator:
    def __init__(self, model_name: str = "fallback", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        require_citations: bool = False,
        allow_no_context: bool = False,
    ) -> str:
        if self._can_use_gemini():
            answer = self._generate_with_gemini(query, retrieved_chunks)
        elif self._can_use_groq():
            answer = self._generate_with_groq(query, retrieved_chunks)
        else:
            answer = self._generate_fallback(query, retrieved_chunks, allow_no_context)

        if require_citations and retrieved_chunks:
            answer = self._append_citations(answer, retrieved_chunks)
        return answer

    def generate_hypothetical_document(self, query: str) -> str:
        if self._can_use_gemini():
            prompt = (
                "Write a short factual passage (6-8 sentences) that could answer the question. "
                "Do not mention that this is hypothetical.\\nQuestion: "
                f"{query}"
            )
            out = self._generate_with_gemini_prompt(prompt)
            if out:
                return out

        if self._can_use_groq():
            prompt = (
                "Write a short factual passage (6-8 sentences) that could answer the question. "
                "Do not mention that this is hypothetical.\\nQuestion: "
                f"{query}"
            )
            out = self._generate_with_groq_prompt(prompt)
            if out:
                return out

        # Deterministic fallback: query-focused pseudo document.
        return (
            f"This passage discusses: {query}. "
            "Key entities, dates, names, and comparative facts related to the question are included "
            "to support retrieval of evidence-bearing passages."
        )

    def _append_citations(self, answer: str, chunks: List[Dict]) -> str:
        lines = []
        used = chunks[: min(3, len(chunks))]
        for i, c in enumerate(used, start=1):
            title = c.get("source_title") or "Unknown source"
            url = c.get("source_url") or ""
            if url:
                lines.append(f"[{i}] {title}. {url}")
            else:
                lines.append(f"[{i}] {title}.")
        return f"{answer}\\n\\nReferences (IEEE style):\\n" + "\\n".join(lines)

    def _build_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, start=1):
            title = c.get("source_title") or "Unknown"
            score = c.get("score", 0.0)
            text = c.get("chunk_text") or ""
            parts.append(f"[{i}] {title} (score={score:.4f}): {text}")
        return "\\n".join(parts)

    def _generate_fallback(self, query: str, chunks: List[Dict], allow_no_context: bool) -> str:
        if not chunks and allow_no_context:
            return "Insufficient reliable context retrieved. The answer cannot be grounded confidently."
        if not chunks:
            return "I could not retrieve enough context to answer confidently."

        # Simple extractive heuristic: return the highest-score chunk sentence that best overlaps with query terms.
        query_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2}

        best_sentence = ""
        best_score = -1
        for c in chunks:
            text = c.get("chunk_text") or ""
            sentences = re.split(r"(?<=[.!?])\\s+", text)
            for sentence in sentences:
                tokens = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                overlap = len(query_terms.intersection(tokens))
                score = overlap + float(c.get("score", 0.0))
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()

        if not best_sentence:
            best_sentence = (chunks[0].get("chunk_text") or "").strip()

        return best_sentence or "No grounded answer found in retrieved context."

    def _can_use_gemini(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY")) and "gemini" in self.model_name.lower()

    def _can_use_groq(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY")) and "gemini" not in self.model_name.lower()

    def _generate_with_gemini(self, query: str, chunks: List[Dict]) -> str:
        prompt = (
            "Answer the question using only the provided context. If unsure, say so briefly.\\n"
            f"Question: {query}\\n\\nContext:\\n{self._build_context(chunks)}"
        )
        out = self._generate_with_gemini_prompt(prompt)
        return out or self._generate_fallback(query, chunks, allow_no_context=False)

    def _generate_with_gemini_prompt(self, prompt: str) -> Optional[str]:
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            return text.strip() if text else None
        except Exception:
            return None

    def _generate_with_groq(self, query: str, chunks: List[Dict]) -> str:
        prompt = (
            "Answer the question using only the provided context. If unsure, say so briefly.\\n"
            f"Question: {query}\\n\\nContext:\\n{self._build_context(chunks)}"
        )
        out = self._generate_with_groq_prompt(prompt)
        return out or self._generate_fallback(query, chunks, allow_no_context=False)

    def _generate_with_groq_prompt(self, prompt: str) -> Optional[str]:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a grounded QA assistant. Use provided evidence only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None
