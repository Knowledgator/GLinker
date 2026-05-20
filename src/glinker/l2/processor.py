from typing import Any, List

from glinker.core.base import BaseProcessor
from glinker.core.registry import processor_registry

from .models import L2Input, L2Config, L2Output, DatabaseRecord
from .component import DatabaseChainComponent


class L2Processor(BaseProcessor[L2Config, L2Input, L2Output]):
    """Multi-layer database search processor."""

    def __init__(
        self,
        config: L2Config,
        component: DatabaseChainComponent,
        pipeline: list[tuple[str, dict[str, Any]]] | None = None,
    ):
        super().__init__(config, component, pipeline)
        self.schema = {}  # Will be set by DAG executor from node config

    def format_label(self, record: DatabaseRecord) -> str:
        """Format label using schema template."""
        template = self.schema.get("template", "{label}")
        try:
            return template.format(**record.model_dump())
        except KeyError:
            return record.label

    def precompute_embeddings(
        self, encoder_fn, target_layers: List[str] | None = None, batch_size: int = 32
    ):
        """
        Precompute embeddings for entities using schema template.

        Args:
            encoder_fn: Function that takes List[str] and returns embeddings
            target_layers: Layer types to update
            batch_size: Batch size for encoding
        """
        template = self.schema.get("template", "{label}")
        model_id = self.config.embeddings.model_name if self.config.embeddings else "unknown"

        return self.component.precompute_embeddings(
            encoder_fn=encoder_fn,
            template=template,
            model_id=model_id,
            target_layers=target_layers,
            batch_size=batch_size,
        )

    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("search", {}),
            ("filter_by_popularity", {}),
            ("deduplicate_candidates", {}),
            ("sort_by_popularity", {}),
            ("limit_candidates", {}),
        ]

    def __call__(
        self,
        mentions: List[str] | List[List[Any]] | L2Input = None,
        texts: List[str] | None = None,
        structure: List[List[str]] | None = None,
        input_data: L2Input = None,
    ) -> L2Output:
        """
        Process mentions and return candidates.

        Supports:
        - List[str]: flat list of mention strings
        - List[List[L1Entity]]: nested list of L1Entity objects (one list per text)
        - L2Input: structured input with mentions and structure
        - mentions=None: return entire entity database (one copy per text)
        """
        if input_data is not None:
            mentions = input_data.mentions
            structure = input_data.structure
        elif isinstance(mentions, L2Input):
            structure = mentions.structure
            mentions = mentions.mentions

        # No mentions → return entire database (simple pipeline mode)
        if mentions is None:
            all_entities = self.component.get_all_entities()
            n = len(texts) if texts is not None else 1
            return L2Output(candidates=[all_entities for _ in range(n)])

        # Check if mentions is nested (list of lists - one per text)
        if mentions and isinstance(mentions[0], (list, tuple)):
            # Nested structure: [[entities_text1], [entities_text2], ...]
            all_candidates = []

            # OPTIMIZATION: Use batch search if available
            if hasattr(self.component, 'batch_search'):
                # Flatten mentions and track structure
                flat_mentions = []
                mention_counts = []

                for text_entities in mentions:
                    mention_texts = [self._extract_mention_text(e) for e in text_entities]
                    flat_mentions.extend(mention_texts)
                    mention_counts.append(len(mention_texts))

                # Batch search all mentions at once
                if flat_mentions:
                    batch_results = self.component.batch_search(flat_mentions)

                    # Apply post-processing pipeline to each result
                    processed_results = []
                    for result in batch_results:
                        # Apply pipeline steps (filter, deduplicate, sort, limit)
                        processed = result
                        for step_name, step_kwargs in self.pipeline:
                            if step_name != "search":  # Skip search, already done
                                method = getattr(self.component, step_name)
                                processed = method(processed, **step_kwargs)
                        processed_results.append(processed)

                    # Reconstruct nested structure
                    idx = 0
                    for count in mention_counts:
                        text_candidates = []
                        for _ in range(count):
                            if idx < len(processed_results):
                                text_candidates.extend(processed_results[idx])
                                idx += 1
                        all_candidates.append(text_candidates)
                else:
                    all_candidates = [[] for _ in mentions]
            else:
                # Fallback: individual searches
                for text_entities in mentions:
                    text_candidates = []

                    for entity in text_entities:
                        # Extract text from L1Entity or dict
                        mention_text = self._extract_mention_text(entity)

                        # Search candidates for this mention
                        candidates = self._execute_pipeline(mention_text, self.pipeline)
                        text_candidates.extend(candidates)

                    all_candidates.append(text_candidates)

            return L2Output(candidates=all_candidates)

        # Flat structure: ["mention1", "mention2", ...]
        else:
            # OPTIMIZATION: Use batch search if available
            if hasattr(self.component, 'batch_search'):
                # Extract mention texts
                mention_texts = [self._extract_mention_text(m) for m in mentions]

                # Batch search all at once
                if mention_texts:
                    batch_results = self.component.batch_search(mention_texts)

                    # Apply post-processing pipeline to each result
                    all_candidates = []
                    for result in batch_results:
                        # Apply pipeline steps (filter, deduplicate, sort, limit)
                        processed = result
                        for step_name, step_kwargs in self.pipeline:
                            if step_name != "search":  # Skip search, already done
                                method = getattr(self.component, step_name)
                                processed = method(processed, **step_kwargs)
                        all_candidates.append(processed)
                else:
                    all_candidates = []
            else:
                # Fallback: individual searches
                all_candidates = []

                for mention in mentions:
                    mention_text = self._extract_mention_text(mention)
                    candidates = self._execute_pipeline(mention_text, self.pipeline)
                    all_candidates.append(candidates)

            if structure:
                grouped = self._group_by_structure(all_candidates, structure)
            else:
                # Flatten all into one group
                grouped = [self._flatten(all_candidates)]

            return L2Output(candidates=grouped)

    def _extract_mention_text(self, mention: Any) -> str:
        """Extract text string from mention (can be L1Entity, dict, or str)."""
        if isinstance(mention, str):
            return mention
        elif hasattr(mention, "text"):
            return mention.text
        elif isinstance(mention, dict):
            return mention.get("text", str(mention))
        else:
            return str(mention)

    def _group_by_structure(
        self, all_candidates: List[List[DatabaseRecord]], structure: List[List[str]]
    ) -> List[List[DatabaseRecord]]:
        """Group candidates according to structure."""
        grouped = []
        idx = 0
        for text_mentions in structure:
            text_candidates = []
            for _ in text_mentions:
                if idx < len(all_candidates):
                    text_candidates.extend(all_candidates[idx])
                    idx += 1
            grouped.append(text_candidates)
        return grouped

    def _flatten(self, nested: List[List[Any]]) -> List[Any]:
        """Flatten nested list."""
        flat = []
        for sublist in nested:
            flat.extend(sublist)
        return flat


@processor_registry.register("l2_chain")
def create_l2_processor(config_dict: dict, pipeline: list | None = None) -> L2Processor:
    """Factory: creates component + processor."""
    config = L2Config(**config_dict)
    component = DatabaseChainComponent(config)
    return L2Processor(config, component, pipeline)
