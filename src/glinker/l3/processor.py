from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, List

import torch

from glinker.core.base import BaseProcessor
from glinker.core.registry import processor_registry

from .models import L3Input, L3Config, L3LLMConfig, L3Entity, L3Output
from .component import L3Component

logger = logging.getLogger(__name__)


class L3Processor(BaseProcessor[L3Config, L3Input, L3Output]):
    """GLiNER entity linking processor."""

    def __init__(
        self,
        config: L3Config,
        component: L3Component,
        pipeline: list[tuple[str, dict[str, Any]]] | None = None,
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
        self.schema = {}
        self._l2_processor = None  # Will be set by DAG executor for cache write-back

    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [("predict_entities", {}), ("sort_by_position", {})]

    def __call__(
        self,
        texts: List[str],
        candidates: List[List[List[Any]]],
        l1_entities: List[List[Any]] | None = None,
        input_data: L3Input = None,
    ) -> L3Output:
        """Process texts with per-span candidate labels.

        Args:
            texts: List of input texts.
            candidates: ``candidates[text_idx][span_idx]`` — L2 candidates per span.
            l1_entities: ``l1_entities[text_idx]`` — L1 entities defining the spans.
        """
        all_entities = []

        for idx, (text, per_span_candidates) in enumerate(zip(texts, candidates)):
            text_l1 = (l1_entities[idx] if l1_entities and idx < len(l1_entities) else [])

            # input_spans from L1 — char-level, aligned with per_span_candidates
            input_spans = [
                {
                    "start": e["start"] if isinstance(e, dict) else e.start,
                    "end": e["end"] if isinstance(e, dict) else e.end,
                }
                for e in text_l1
            ]

            # Build flat labels + per-span indices from L2 per-span candidates
            labels, span_label_indices, label_to_candidate = self._build_sparse_inputs(
                per_span_candidates
            )

            if not labels:
                all_entities.append([])
                continue

            use_precomputed = (
                self.config.use_precomputed_embeddings
                and self.component.supports_precomputed_embeddings
                and self._can_use_precomputed(
                    [c for span in per_span_candidates for c in span],
                    label_to_candidate,
                )
            )

            if use_precomputed:
                flat_candidates = [c for span in per_span_candidates for c in span]
                embeddings = self._get_embeddings_tensor(
                    flat_candidates, labels, label_to_candidate
                )
                entities = self.component.predict_with_embeddings(
                    text, labels, embeddings, input_spans, span_label_indices
                )
            elif self.component.supports_precomputed_embeddings:
                # Encode in chunks (encode_labels defaults to batch_size=32) so we
                # never push thousands of labels through DeBERTa in a single pass → OOM.
                embeddings = self.component.encode_labels(labels)
                entities = self.component.predict_with_embeddings(
                    text, labels, embeddings, input_spans, span_label_indices
                )
                if self.config.cache_embeddings:
                    flat_candidates = [c for span in per_span_candidates for c in span]
                    # Pass the tensor we just computed — skip a second encode_labels call.
                    self._cache_embeddings_tensor(
                        flat_candidates, labels, label_to_candidate, embeddings
                    )
            else:
                entities = self.component.predict_entities(
                    text, labels, input_spans, span_label_indices
                )

            # Apply rest of pipeline (sort, dedup, etc.)
            for method_name, kwargs in self.pipeline[1:]:
                method = getattr(self.component, method_name)
                entities = method(entities, **kwargs)

            if self.schema.get("ranking"):
                flat_candidates = [c for span in per_span_candidates for c in span]
                entities = self._rank_entities(entities, flat_candidates)

            all_entities.append(entities)

        return L3Output(entities=all_entities)

    def _build_sparse_inputs(
        self,
        per_span_candidates: List[List[Any]],
    ) -> tuple[List[str], List[List[int]], dict]:
        """Build flat labels list and per-span label indices from L2 per-span candidates.

        Args:
            per_span_candidates: ``per_span_candidates[span_idx]`` = list of candidates.

        Returns:
            ``(labels, span_label_indices, label_to_candidate)``
            - labels: flat deduplicated label strings
            - span_label_indices: for each span, indices into labels
            - label_to_candidate: label string → candidate object
        """
        template = self.schema.get("template", "{label}")
        label_to_idx: dict = {}
        labels: List[str] = []
        label_to_candidate: dict = {}
        span_label_indices: List[List[int]] = []

        for span_cands in per_span_candidates:
            idxs: List[int] = []
            for cand in span_cands:
                lbl = self._format_label(cand, template)
                if lbl not in label_to_idx:
                    label_to_idx[lbl] = len(labels)
                    labels.append(lbl)
                    label_to_candidate[lbl] = cand
                idxs.append(label_to_idx[lbl])
            span_label_indices.append(idxs)

        return labels, span_label_indices, label_to_candidate

    def _format_label(self, candidate: Any, template: str) -> str:
        """Format a candidate into a label string using the schema template."""
        try:
            if hasattr(candidate, "model_dump"):
                return template.format(**candidate.model_dump())
            if hasattr(candidate, "dict"):
                return template.format(**candidate.dict())
            if isinstance(candidate, dict):
                return template.format(**candidate)
        except KeyError:
            pass
        return getattr(candidate, "label", str(candidate))

    def _can_use_precomputed(self, candidates: List[Any], label_to_candidate: dict) -> bool:
        """Check if all candidates have compatible precomputed embeddings."""
        if not candidates:
            return False

        expected_model = self.config.model_name

        for candidate in candidates:
            # Check if candidate has embedding
            embedding = getattr(candidate, "embedding", None)
            if embedding is None:
                return False

            # Check if model matches
            model_id = getattr(candidate, "embedding_model_id", None)
            if model_id != expected_model:
                return False

        return True

    def _get_embeddings_tensor(
        self, candidates: List[Any], labels: List[str], label_to_candidate: dict
    ) -> torch.Tensor:
        """Build embeddings tensor from candidates in same order as labels."""
        embeddings = []

        for label in labels:
            candidate = label_to_candidate.get(label)
            if candidate and hasattr(candidate, "embedding") and candidate.embedding:
                embeddings.append(candidate.embedding)
            else:
                # Should not happen if _can_use_precomputed returned True
                raise ValueError(f"Missing embedding for label: {label}")

        return torch.tensor(embeddings, device=self.component.device)

    def _cache_embeddings(self, candidates: List[Any], labels: List[str], label_to_candidate: dict):
        """Compute and cache embeddings for candidates without them."""
        if not self._l2_processor:
            return

        # Find candidates without embeddings
        to_compute = []
        to_compute_ids = []

        for candidate in candidates:
            if not getattr(candidate, "embedding", None):
                to_compute.append(candidate)
                to_compute_ids.append(candidate.entity_id)

        if not to_compute:
            return

        # Format labels for these candidates
        template = self.schema.get("template", "{label}")
        compute_labels = [self._format_label(c, template) for c in to_compute]

        # Encode labels
        embeddings = self.component.encode_labels(compute_labels)

        # Update L2 layer
        if hasattr(self._l2_processor, "component"):
            for layer in self._l2_processor.component.layers:
                if layer.is_available():
                    layer.update_embeddings(
                        to_compute_ids, embeddings.tolist(), self.config.model_name
                    )
                    break  # Update first available layer

    def _cache_embeddings_tensor(
        self,
        candidates: List[Any],
        labels: List[str],
        label_to_candidate: dict,
        embeddings: "torch.Tensor",
    ):
        """Cache pre-computed embeddings for candidates that don't have them yet.

        Unlike :meth:`_cache_embeddings`, accepts an already-encoded tensor so
        we skip a redundant ``encode_labels`` call when the caller just encoded.
        ``embeddings[i]`` must correspond to ``labels[i]``.
        """
        if not self._l2_processor:
            return

        label_to_emb_idx = {lbl: i for i, lbl in enumerate(labels)}
        template = self.schema.get("template", "{label}")

        to_update_ids: List[Any] = []
        to_update_vecs: List[Any] = []

        for candidate in candidates:
            if getattr(candidate, "embedding", None):
                continue  # already cached
            lbl = self._format_label(candidate, template)
            emb_idx = label_to_emb_idx.get(lbl)
            if emb_idx is None:
                continue
            to_update_ids.append(candidate.entity_id)
            to_update_vecs.append(embeddings[emb_idx].tolist())

        if not to_update_ids:
            return

        if hasattr(self._l2_processor, "component"):
            for layer in self._l2_processor.component.layers:
                if layer.is_available():
                    layer.update_embeddings(
                        to_update_ids, to_update_vecs, self.config.model_name
                    )
                    break

    def _rank_entities(self, entities: List[L3Entity], candidates: List[Any]) -> List[L3Entity]:
        """Re-rank entities using multiple scoring factors."""
        # Build label to candidate mapping
        label_to_candidate = {}
        for c in candidates:
            if hasattr(c, "label"):
                label_to_candidate[c.label] = c
                if hasattr(c, "aliases"):
                    for alias in c.aliases:
                        if alias not in label_to_candidate:
                            label_to_candidate[alias] = c

        # Calculate weighted scores
        for entity in entities:
            total_score = 0.0
            total_weight = 0.0

            for rank_spec in self.schema["ranking"]:
                field = rank_spec["field"]
                weight = rank_spec["weight"]
                total_weight += weight

                if field == "gliner_score":
                    total_score += entity.score * weight
                else:
                    candidate = label_to_candidate.get(entity.label)
                    if candidate and hasattr(candidate, field):
                        value = getattr(candidate, field, 0)
                        if isinstance(value, (int, float)):
                            normalized = min(value / 1000000.0, 1.0)
                            total_score += normalized * weight

            if total_weight > 0:
                entity.score = total_score / total_weight

        return sorted(entities, key=lambda x: x.score, reverse=True)


@processor_registry.register("l3_batch")
def create_l3_processor(config_dict: dict, pipeline: list | None = None) -> L3Processor:
    """Factory: creates component + processor."""
    config = L3Config(**config_dict)
    component = L3Component(config)
    return L3Processor(config, component, pipeline)


# ── LLM-based L3 ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an entity disambiguation assistant.
For each mention you will receive:
  - The sentence with the mention wrapped in [[ ]].
  - A numbered list of candidate entities (label: description).

Reply with ONLY a JSON array, one entry per mention, in the same order.
Each entry must be either:
  - an integer index (0-based) of the correct candidate, or
  - null if none of the candidates fits.

Example reply: [0, null, 2]
Do NOT add any explanation or extra text."""


def _mark_mention(text: str, start: int, end: int) -> str:
    return text[:start] + "[[" + text[start:end] + "]]" + text[end:]


def _build_prompt(batch: list[dict]) -> str:
    lines = []
    for i, item in enumerate(batch):
        lines.append(f"Mention {i}: {item['context']}")
        for j, c in enumerate(item["candidates"]):
            desc = c.get("description") or ""
            label = c.get("label", "")
            lines.append(f"  {j}. {label}: {desc}" if desc else f"  {j}. {label}")
        lines.append("")
    return "\n".join(lines)


def _parse_llm_response(raw: str, expected: int) -> list[int | None]:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        logger.warning(f"No JSON array in LLM response: {raw!r}")
        return [None] * expected
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}. Raw: {raw!r}")
        return [None] * expected
    result = []
    for i in range(expected):
        val = parsed[i] if i < len(parsed) else None
        result.append(int(val) if isinstance(val, (int, float)) and val == int(val) else None)
    return result


class L3LLMProcessor:
    """LLM-based entity disambiguation — drop-in replacement for L3Processor."""

    def __init__(self, config: L3LLMConfig):
        self.config = config
        self.schema = {}
        self._client = self._build_client()

    def _build_client(self):
        if self.config.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("pip install google-generativeai")
            api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY env var or config.api_key")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(
                self.config.model_name,
                system_instruction=_SYSTEM_PROMPT,
            )
        elif self.config.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            return OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider!r}")

    def _complete(self, prompt: str) -> str:
        if self.config.provider == "gemini":
            response = self._client.generate_content(
                prompt,
                generation_config={"temperature": self.config.temperature},
            )
            return response.text
        else:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content

    def _call_with_retry(self, batch_items: list[dict]) -> list[int | None]:
        prompt = _build_prompt(batch_items)
        for attempt in range(self.config.max_retries):
            try:
                raw = self._complete(prompt)
                return _parse_llm_response(raw, len(batch_items))
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        return [None] * len(batch_items)

    def _cand_info(self, c: Any) -> dict:
        get = lambda attr: getattr(c, attr, c.get(attr, "") if isinstance(c, dict) else "")
        return {"label": get("label"), "description": get("description"), "entity_id": get("entity_id")}

    def __call__(
        self,
        texts: List[str],
        candidates: List[List[List[Any]]],
        l1_entities: List[List[Any]] | None = None,
        input_data: Any = None,
    ) -> L3Output:
        all_entities: List[List[L3Entity]] = []

        for text_idx, (text, per_span_candidates) in enumerate(zip(texts, candidates)):
            l1 = l1_entities[text_idx] if l1_entities and text_idx < len(l1_entities) else []

            items = []
            for entity, cands in zip(l1, per_span_candidates):
                if not cands:
                    items.append(None)
                    continue
                start = entity["start"] if isinstance(entity, dict) else entity.start
                end = entity["end"] if isinstance(entity, dict) else entity.end
                top_cands = cands[: self.config.top_k]
                items.append({
                    "start": start,
                    "end": end,
                    "mention_text": text[start:end],
                    "context": _mark_mention(text, start, end),
                    "candidates": [self._cand_info(c) for c in top_cands],
                    "raw_candidates": top_cands,
                })

            non_null = [(i, item) for i, item in enumerate(items) if item is not None]
            choices: dict[int, int | None] = {}

            for batch_start in range(0, len(non_null), self.config.batch_size):
                batch = non_null[batch_start: batch_start + self.config.batch_size]
                results = self._call_with_retry([item for _, item in batch])
                for local_idx, choice in enumerate(results):
                    choices[batch[local_idx][0]] = choice

            text_entities: List[L3Entity] = []
            for i, item in enumerate(items):
                if item is None:
                    continue
                choice = choices.get(i)
                if choice is None or not (0 <= choice < len(item["raw_candidates"])):
                    continue
                cand = item["raw_candidates"][choice]
                label = getattr(cand, "label", cand.get("label", "") if isinstance(cand, dict) else "")
                text_entities.append(L3Entity(
                    text=item["mention_text"],
                    label=label,
                    start=item["start"],
                    end=item["end"],
                    score=1.0,
                ))

            all_entities.append(text_entities)

        return L3Output(entities=all_entities)


@processor_registry.register("l3_llm")
def create_l3_llm_processor(config_dict: dict, pipeline: list | None = None) -> L3LLMProcessor:
    config = L3LLMConfig(**config_dict)
    return L3LLMProcessor(config)
