from typing import List, Optional, Dict, Tuple
from glinker.core.base import BaseComponent
from .models import (
    L0Config, L0Entity, LinkedEntity
)
from glinker.l1.models import L1Entity
from glinker.l2.models import DatabaseRecord
from glinker.l3.models import L3Entity


class L0Component(BaseComponent[L0Config]):
    """
    L0 aggregation component - combines outputs from L1, L2, L3

    Workflow:
    1. For each L1 mention → find its L2 candidates
    2. For each L1 mention → check if it was linked in L3
    3. Create L0Entity with full information from all layers
    """

    def get_available_methods(self) -> List[str]:
        return [
            "aggregate",
            "filter_by_confidence",
            "sort_by_confidence",
            "calculate_stats"
        ]

    def aggregate(
        self,
        l1_entities: List[List[L1Entity]],
        l2_candidates: List[List[DatabaseRecord]],
        l3_entities: List[List[L3Entity]],
        template: str = "{label}"
    ) -> List[List[L0Entity]]:
        """
        Main aggregation method - combines all layers

        Args:
            l1_entities: [[L1Entity, ...], ...] - one list per text
            l2_candidates: [[DatabaseRecord, ...], ...] - one list per text
            l3_entities: [[L3Entity, ...], ...] - one list per text
            template: Label template from L3 schema (e.g., "{label} {description}")

        Returns:
            [[L0Entity, ...], ...] - aggregated entities per text
        """
        all_results = []

        # Process each text separately
        for text_idx in range(len(l1_entities)):
            l1_mentions = l1_entities[text_idx] if text_idx < len(l1_entities) else []
            l2_cands = l2_candidates[text_idx] if text_idx < len(l2_candidates) else []
            l3_links = l3_entities[text_idx] if text_idx < len(l3_entities) else []

            text_results = self._aggregate_single_text(l1_mentions, l2_cands, l3_links, template)
            all_results.append(text_results)

        return all_results

    def _aggregate_single_text(
        self,
        l1_mentions: List[L1Entity],
        l2_candidates: List[DatabaseRecord],
        l3_links: List[L3Entity],
        template: str = "{label}"
    ) -> List[L0Entity]:
        """
        Aggregate data for a single text

        Strategy:
        1. Build index of L3 linked entities by position
        2. For each L1 mention:
           - Find corresponding candidates from L2
           - Check if it was linked in L3
           - Create L0Entity
        3. If strict_matching=False, also include L3 entities outside L1 mentions
        """
        # Build L3 index by position
        l3_by_position = self._build_l3_index(l3_links)

        results = []
        used_l3_positions = set()  # Track which L3 entities were matched to L1 mentions

        for mention_idx, l1_mention in enumerate(l1_mentions):
            # Get candidates for this mention (L2 returns flat list, need to group)
            # Assuming L2 candidates are in the same order as L1 mentions
            mention_candidates = self._get_candidates_for_mention(
                mention_idx, l1_mention, l2_candidates
            )

            # Check if this mention was linked in L3
            linked_entity, l3_pos = self._find_linked_entity_with_position(
                l1_mention, l3_by_position, mention_candidates, template,
                tolerance=self.config.position_tolerance
            )

            if l3_pos:
                used_l3_positions.add(l3_pos)

            # Build candidate_scores from L3 class_probs
            candidate_scores = {}
            l3_entity = l3_by_position.get(l3_pos) if l3_pos else None
            if l3_entity and l3_entity.class_probs:
                candidate_scores = self._build_candidate_scores(
                    l3_entity.class_probs, l2_candidates, template
                )

            # Determine pipeline stage
            pipeline_stage = self._determine_stage(mention_candidates, linked_entity)

            # Create L0Entity
            l0_entity = L0Entity(
                mention_text=l1_mention.text,
                mention_start=l1_mention.start,
                mention_end=l1_mention.end,
                left_context=l1_mention.left_context,
                right_context=l1_mention.right_context,
                candidates=mention_candidates,
                num_candidates=len(mention_candidates),
                linked_entity=linked_entity,
                is_linked=linked_entity is not None,
                candidate_scores=candidate_scores,
                pipeline_stage=pipeline_stage
            )

            results.append(l0_entity)

        # If loose mode, include L3 entities that weren't matched to L1 mentions
        if not self.config.strict_matching:
            for (l3_start, l3_end), l3_entity in l3_by_position.items():
                if (l3_start, l3_end) not in used_l3_positions:
                    # This L3 entity was not matched to any L1 mention
                    # Find candidate by label
                    matched_candidate = self._match_candidate_by_label(
                        l3_entity.label, l2_candidates, template
                    )

                    # Build candidate_scores from class_probs
                    candidate_scores = {}
                    if l3_entity.class_probs:
                        candidate_scores = self._build_candidate_scores(
                            l3_entity.class_probs, l2_candidates, template
                        )

                    linked = LinkedEntity(
                        entity_id=matched_candidate.entity_id if matched_candidate else "unknown",
                        label=matched_candidate.label if matched_candidate else l3_entity.label,
                        confidence=l3_entity.score,
                        start=l3_entity.start,
                        end=l3_entity.end,
                        matched_text=l3_entity.text
                    )

                    l0_entity = L0Entity(
                        mention_text=l3_entity.text,
                        mention_start=l3_entity.start,
                        mention_end=l3_entity.end,
                        left_context="",  # No context from L1
                        right_context="",
                        candidates=[matched_candidate] if matched_candidate else [],
                        num_candidates=1 if matched_candidate else 0,
                        linked_entity=linked,
                        is_linked=True,
                        candidate_scores=candidate_scores,
                        pipeline_stage="l3_only"  # Indicates L3 found it without L1
                    )
                    results.append(l0_entity)

        return results

    def _build_l3_index(self, l3_links: List[L3Entity]) -> Dict[Tuple[int, int], L3Entity]:
        """Build index of L3 entities by (start, end) position"""
        index = {}
        for entity in l3_links:
            key = (entity.start, entity.end)
            index[key] = entity
        return index

    def _get_candidates_for_mention(
        self,
        mention_idx: int,
        l1_mention: L1Entity,
        all_candidates: List[DatabaseRecord]
    ) -> List[DatabaseRecord]:
        """
        Get candidates for specific mention

        Note: L2 returns candidates grouped per text. We need to match by text content.
        """
        matched_candidates = []

        # Match candidates by mention text (normalize)
        mention_text_lower = l1_mention.text.lower().strip()

        for candidate in all_candidates:
            # Check if candidate matches this mention
            if candidate.label.lower().strip() == mention_text_lower:
                matched_candidates.append(candidate)
                continue

            # Check aliases
            for alias in candidate.aliases:
                if alias.lower().strip() == mention_text_lower:
                    matched_candidates.append(candidate)
                    break

        return matched_candidates

    def _find_linked_entity(
        self,
        l1_mention: L1Entity,
        l3_by_position: Dict[Tuple[int, int], L3Entity],
        candidates: List[DatabaseRecord],
        template: str = "{label}"
    ) -> Optional[LinkedEntity]:
        """
        Find if this L1 mention was linked in L3

        Strategy:
        1. Look up L3 entity by position (start, end)
        2. If found, match with candidates to get entity_id
        3. Return LinkedEntity with full information
        """
        linked, _ = self._find_linked_entity_with_position(
            l1_mention, l3_by_position, candidates, template
        )
        return linked

    def _find_linked_entity_with_position(
        self,
        l1_mention: L1Entity,
        l3_by_position: Dict[Tuple[int, int], L3Entity],
        candidates: List[DatabaseRecord],
        template: str = "{label}",
        tolerance: int = 2
    ) -> Tuple[Optional[LinkedEntity], Optional[Tuple[int, int]]]:
        """
        Find if this L1 mention was linked in L3, and return the matched position

        Returns:
            Tuple of (LinkedEntity or None, matched position tuple or None)
        """
        # Try exact position match
        key = (l1_mention.start, l1_mention.end)
        l3_entity = l3_by_position.get(key)
        matched_key = key if l3_entity else None

        if not l3_entity:
            # Try fuzzy position match (text might be slightly different)
            l3_entity, matched_key = self._fuzzy_position_match_with_key(
                l1_mention.start, l1_mention.end, l3_by_position, tolerance
            )

        if not l3_entity:
            return None, None

        # Find matching candidate by label using template
        matched_candidate = self._match_candidate_by_label(l3_entity.label, candidates, template)

        if not matched_candidate:
            # L3 found entity but no matching candidate - shouldn't happen but handle gracefully
            return LinkedEntity(
                entity_id="unknown",
                label=l3_entity.label,
                confidence=l3_entity.score,
                start=l3_entity.start,
                end=l3_entity.end,
                matched_text=l3_entity.text
            ), matched_key

        return LinkedEntity(
            entity_id=matched_candidate.entity_id,
            label=matched_candidate.label,
            confidence=l3_entity.score,
            start=l3_entity.start,
            end=l3_entity.end,
            matched_text=l3_entity.text
        ), matched_key

    def _fuzzy_position_match(
        self,
        start: int,
        end: int,
        l3_by_position: Dict[Tuple[int, int], L3Entity],
        tolerance: int = 2
    ) -> Optional[L3Entity]:
        """Find L3 entity with position close to given range"""
        entity, _ = self._fuzzy_position_match_with_key(start, end, l3_by_position, tolerance)
        return entity

    def _fuzzy_position_match_with_key(
        self,
        start: int,
        end: int,
        l3_by_position: Dict[Tuple[int, int], L3Entity],
        tolerance: int = 2
    ) -> Tuple[Optional[L3Entity], Optional[Tuple[int, int]]]:
        """Find L3 entity with position close to given range, return with its key"""
        for (l3_start, l3_end), entity in l3_by_position.items():
            if abs(l3_start - start) <= tolerance and abs(l3_end - end) <= tolerance:
                return entity, (l3_start, l3_end)
        return None, None

    def _build_candidate_scores(
        self,
        class_probs: Dict[str, float],
        candidates: List[DatabaseRecord],
        template: str = "{label}"
    ) -> Dict[str, float]:
        """
        Map L3 class_probs (label -> probability) to candidate entity_ids.

        Args:
            class_probs: Dict of label -> probability from L3 entity
            candidates: L2 candidate records
            template: Schema template used to format labels in L3

        Returns:
            Dict of entity_id -> probability
        """
        scores = {}
        for label, prob in class_probs.items():
            matched = self._match_candidate_by_label(label, candidates, template)
            if matched:
                scores[matched.entity_id] = prob
        return scores

    def _match_candidate_by_label(
        self,
        l3_label: str,
        candidates: List[DatabaseRecord],
        template: str = "{label}"
    ) -> Optional[DatabaseRecord]:
        """
        Match L3 label with L2 candidate using the same template

        Uses the schema template to format candidate labels the same way L3 did,
        enabling exact matching.

        Example:
            template = "{label} {description}"
            L3 label = "TP53 Tumor suppressor gene..."
            candidate formatted = "TP53 Tumor suppressor gene..." -> MATCH!

        Args:
            l3_label: Label from L3 entity (formatted with template)
            candidates: List of candidates from L2
            template: Template string (e.g., "{label} {description}")

        Returns:
            Matched DatabaseRecord or None
        """
        l3_label_lower = l3_label.lower().strip()

        # Try to match by formatting each candidate with the template
        for candidate in candidates:
            try:
                # Format candidate using same template as L3
                if hasattr(candidate, 'dict'):
                    cand_dict = candidate.dict()
                else:
                    cand_dict = {
                        'label': candidate.label,
                        'description': getattr(candidate, 'description', ''),
                        'entity_id': getattr(candidate, 'entity_id', ''),
                        'entity_type': getattr(candidate, 'entity_type', ''),
                        'popularity': getattr(candidate, 'popularity', 0),
                        'aliases': getattr(candidate, 'aliases', [])
                    }

                formatted_label = template.format(**cand_dict)

                if formatted_label.lower().strip() == l3_label_lower:
                    return candidate

            except (KeyError, AttributeError):
                # Template formatting failed, try simple label match
                if candidate.label.lower().strip() == l3_label_lower:
                    return candidate

        # Fallback: try simple contains match (for robustness)
        for candidate in candidates:
            cand_label_lower = candidate.label.lower().strip()
            if cand_label_lower and cand_label_lower in l3_label_lower:
                return candidate

        return None

    def _determine_stage(
        self,
        candidates: List[DatabaseRecord],
        linked_entity: Optional[LinkedEntity]
    ) -> str:
        """Determine which pipeline stage was last successful"""
        if linked_entity:
            return "l3_linked"
        elif candidates:
            return "l2_found"
        else:
            return "l1_only"

    def filter_by_confidence(
        self,
        entities: List[List[L0Entity]],
        min_confidence: float = None
    ) -> List[List[L0Entity]]:
        """Filter entities by linking confidence"""
        threshold = min_confidence if min_confidence is not None else self.config.min_confidence

        filtered = []
        for text_entities in entities:
            filtered_text = [
                e for e in text_entities
                if e.linked_entity and e.linked_entity.confidence >= threshold
            ]
            filtered.append(filtered_text)

        return filtered

    def sort_by_confidence(self, entities: List[List[L0Entity]]) -> List[List[L0Entity]]:
        """Sort entities by linking confidence (descending)"""
        sorted_results = []
        for text_entities in entities:
            sorted_text = sorted(
                text_entities,
                key=lambda e: e.linked_entity.confidence if e.linked_entity else 0.0,
                reverse=True
            )
            sorted_results.append(sorted_text)
        return sorted_results

    def calculate_stats(self, entities: List[List[L0Entity]]) -> dict:
        """Calculate pipeline statistics"""
        total = 0
        linked = 0
        unlinked = 0
        l1_only = 0
        l2_found = 0
        l3_linked = 0
        l3_only = 0  # L3 entities found without L1 mentions (loose mode)

        for text_entities in entities:
            for entity in text_entities:
                total += 1

                if entity.is_linked:
                    linked += 1
                else:
                    unlinked += 1

                if entity.pipeline_stage == "l1_only":
                    l1_only += 1
                elif entity.pipeline_stage == "l2_found":
                    l2_found += 1
                elif entity.pipeline_stage == "l3_linked":
                    l3_linked += 1
                elif entity.pipeline_stage == "l3_only":
                    l3_only += 1

        return {
            "total_mentions": total,
            "linked": linked,
            "unlinked": unlinked,
            "linking_rate": linked / total if total > 0 else 0.0,
            "stages": {
                "l1_only": l1_only,
                "l2_found": l2_found,
                "l3_linked": l3_linked,
                "l3_only": l3_only
            }
        }
