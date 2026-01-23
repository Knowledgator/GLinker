from typing import Any, List
from src.core.base import BaseProcessor
from src.core.registry import processor_registry
from .models import L0Config, L0Input, L0Output, L0Entity
from .component import L0Component
from src.l1.models import L1Entity
from src.l2.models import DatabaseRecord
from src.l3.models import L3Entity


class L0Processor(BaseProcessor[L0Config, L0Input, L0Output]):
    """
    L0 aggregation processor - combines outputs from all pipeline layers

    This processor aggregates information from:
    - L1: Entity mentions (text, position, context)
    - L2: Candidate entities (database records)
    - L3: Linked entities (disambiguation results)

    Into a unified L0Entity structure showing the full pipeline flow.
    """

    def __init__(
        self,
        config: L0Config,
        component: L0Component,
        pipeline: list[tuple[str, dict[str, Any]]] = None
    ):
        super().__init__(config, component, pipeline)
        self._validate_pipeline()
        self.schema = {}  # Will be set by DAG executor if node has schema

    def _default_pipeline(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("aggregate", {}),
            ("filter_by_confidence", {}),
            ("sort_by_confidence", {}),
            ("calculate_stats", {})
        ]

    def __call__(
        self,
        l1_entities: List[List[L1Entity]] = None,
        l2_candidates: List[List[DatabaseRecord]] = None,
        l3_entities: List[List[L3Entity]] = None,
        input_data: L0Input = None
    ) -> L0Output:
        """
        Process and aggregate outputs from L1, L2, L3

        Args:
            l1_entities: Entities from L1 (mention extraction)
            l2_candidates: Candidates from L2 (database search)
            l3_entities: Entities from L3 (entity linking)
            input_data: Alternative: L0Input with all data

        Returns:
            L0Output with aggregated entities and statistics
        """

        # Support both direct params and L0Input
        if input_data is not None:
            l1_entities = input_data.l1_entities
            l2_candidates = input_data.l2_candidates
            l3_entities = input_data.l3_entities
        elif l1_entities is None or l2_candidates is None or l3_entities is None:
            raise ValueError(
                "Either provide 'l1_entities', 'l2_candidates', 'l3_entities' "
                "or 'input_data'"
            )

        # Pass schema template to component for matching
        template = self.schema.get('template', '{label}') if self.schema else '{label}'

        # Execute aggregation pipeline
        aggregated_entities = self.component.aggregate(
            l1_entities, l2_candidates, l3_entities, template=template
        )

        # Apply pipeline transformations (filter, sort, etc.)
        results = aggregated_entities
        stats = {}

        for method_name, kwargs in self.pipeline[1:]:  # Skip 'aggregate' as we already did it
            if method_name == "calculate_stats":
                stats = self.component.calculate_stats(results)
            else:
                method = getattr(self.component, method_name)
                results = method(results, **kwargs)

        # Calculate final stats if not already done
        if not stats:
            stats = self.component.calculate_stats(results)

        return L0Output(entities=results, stats=stats)


@processor_registry.register("l0_aggregator")
def create_l0_processor(config_dict: dict, pipeline: list = None) -> L0Processor:
    """Factory: creates component + processor"""
    config = L0Config(**config_dict)
    component = L0Component(config)
    return L0Processor(config, component, pipeline)
