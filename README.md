# GLiNKER - Entity Linking Framework
<div align="center">
    <div>
        <a href="https://arxiv.org/abs/2406.12925"><img src="https://img.shields.io/badge/arXiv-2406.12925-b31b1b.svg" alt="GLiNER-bi-Encoder"></a>
        <a href="https://discord.gg/HbW9aNJ9"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
        <a href="https://github.com/Knowledgator/EntityLinker/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Knowledgator/EntityLinker?color=blue"></a>
        <a href="https://hf.co/collections/knowledgator/gliner-linker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="HuggingFace Models"></a>
        <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
        <a href="https://pypi.org/project/glinker/"><img src="https://badge.fury.io/py/glinker.svg" alt="PyPI version"></a>
    </div>
    <br>
</div>

![alt text](logo/header.png)
<!-- ![alt text](image-1.png) -->
> A modular, production-ready entity linking framework combining NER, multi-layer database search, and neural entity disambiguation.

## Overview
GLiNKER is a 4-layer entity linking pipeline that transforms raw text into structured, disambiguated entity mentions. It's designed for:
- **Production use**: Multi-layer caching (Redis → Elasticsearch → PostgreSQL)
- **Research flexibility**: Fully configurable YAML pipelines
- **Performance**: Embedding precomputation for BiEncoder models
- **Scalability**: DAG-based execution with batch processing

### Why GLiNKER?

```python
# Traditional approach: Complex, coupled code
ner_results = spacy_model(text)
candidates = search_database(ner_results)
linked = gliner_model.disambiguate(candidates)
# Mix of models, databases, and business logic

# GLiNKER approach: Declarative configuration
from glinker import ConfigBuilder, DAGExecutor

builder = ConfigBuilder(name="biomedical_el")
builder.l1.gliner(model="knowledgator/gliner-bi-base-v2.0", labels=["gene", "protein", "disease"])
builder.l2.add("redis", priority=2).add("postgres", priority=0)
builder.l3.configure(model="knowledgator/gliner-linker-large-v1.0")

executor = DAGExecutor(builder.get_config())
result = executor.execute({"texts": ["TP53 mutations cause cancer"]})
```

## Quick Start

### Installation

```bash
# From source
git clone https://github.com/Knowledgator/EntityLinker.git
cd EntityLinker
pip install -e .

# With optional dependencies
pip install -e ".[dev,demo]"
```

### 30-Second Example

```python
from glinker import ConfigBuilder, DAGExecutor

# 1. Build configuration
builder = ConfigBuilder(name="demo")
builder.l1.spacy(model="en_core_sci_sm")
builder.l3.configure(model="knowledgator/gliner-linker-large-v1.0")

# 2. Create executor
executor = DAGExecutor(builder.get_config())

# 3. Load entities
executor.load_entities("data/entities.jsonl", target_layers=["dict"])

# 4. Process text
result = executor.execute({
    "texts": ["BRCA1 mutations are associated with breast cancer."]
})

# 5. Get results
l0_result = result.get("l0_result")
for entity in l0_result.entities:
    if entity.linked_entity:
        print(f"{entity.mention_text} → {entity.linked_entity.label}")
        print(f"  Confidence: {entity.linked_entity.score:.3f}")
```

**Output:**
```
BRCA1 → BRCA1: Breast cancer type 1 susceptibility protein
  Confidence: 0.923
breast cancer → Breast Cancer: Malignant neoplasm of the breast
  Confidence: 0.887
```

## Features

### Multiple NER Backends
- **spaCy** - Fast, rule-based NER for standard use cases
- **GLiNER** - Neural NER with custom labels (no training required)

### Multi-Layer Database Support
- **Dict** - In-memory (perfect for demos)
- **Redis** - Fast cache (production)
- **Elasticsearch** - Full-text search with fuzzy matching
- **PostgreSQL** - Persistent storage with pg_trgm fuzzy search

### Performance Optimization
- **Embedding Precomputation** - Cache label embeddings for BiEncoder models
- **Cache Hierarchy** - Automatic write-back: Redis → ES → PostgreSQL
- **Batch Processing** - Efficient parallel processing

### Configuration Builders

**Simple Usage (Auto-defaults):**
```python
builder = ConfigBuilder(name="simple")
builder.l1.gliner(model="...", labels=["gene", "protein"])
builder.l3.configure(model="...")
builder.save("config.yaml")  # Dict layer added automatically
```

**Advanced Usage (Full Control):**
```python
builder = ConfigBuilder(name="production")
builder.l1.gliner(model="...", labels=["gene", "protein"], use_precomputed_embeddings=True)
builder.l2.add("redis", priority=2, ttl=3600)
builder.l2.add("elasticsearch", priority=1, ttl=86400)
builder.l2.add("postgres", priority=0)
builder.l2.embeddings(enabled=True, model_name="...")
builder.l3.configure(model="...", use_precomputed_embeddings=True)
builder.l0.configure(strict_matching=True, min_confidence=0.3)
builder.save("config.yaml")
```

## Architecture

GLiNKER uses a **4-layer pipeline**:

![alt text](logo/architecture.png)


**Key Concepts:**

- **DAG Execution**: Layers execute in dependency order with automatic data flow
- **Component-Processor Pattern**: Each layer has a Component (methods) and Processor (orchestration)
- **Schema Consistency**: Single template (e.g., `"{label}: {description}"`) across layers
- **Cache Hierarchy**: Upper layers cache results from lower layers automatically

## Use Cases

### Biomedical Text Mining
```python
builder.l1.gliner(
    model="knowledgator/gliner-bi-base-v2.0",
    labels=["gene", "protein", "disease", "drug", "chemical"]
)
```

### News Article Analysis
```python
builder.l1.spacy(model="en_core_web_lg")
# Link to Wikidata/Wikipedia entities
```

### Clinical NLP
```python
builder.l1.gliner(
    model="knowledgator/gliner-bi-base-v2.0",
    labels=["symptom", "diagnosis", "medication", "procedure"]
)
```

## Database Setup

### Quick Start (Docker)
```bash
# Start all databases
cd scripts/database
docker-compose up -d

# Load entities
python scripts/database/setup_all.sh
```

### Manual Setup
```python
from glinker import DAGExecutor

executor = DAGExecutor(pipeline)
executor.load_entities(
    filepath="data/entities.jsonl",
    target_layers=["redis", "elasticsearch", "postgres"],
    batch_size=1000
)
```

**Entity Format (JSONL):**
```json
{"entity_id": "Q123", "label": "BRCA1", "description": "Breast cancer gene", "entity_type": "gene", "popularity": 1000000, "aliases": ["BRCA-1"]}
{"entity_id": "Q456", "label": "TP53", "description": "Tumor protein p53", "entity_type": "gene", "popularity": 950000, "aliases": ["p53"]}
```

## Configuration Examples

### In-Memory (Development)
```yaml
# configs/pipelines/dict/default.yaml
name: "dev_pipeline"
nodes:
  - id: "l1"
    processor: "l1_spacy"
    config:
      model: "en_core_sci_sm"

  - id: "l2"
    processor: "l2_chain"
    config:
      layers:
        - type: "dict"
          search_mode: ["exact", "fuzzy"]

  - id: "l3"
    processor: "l3_batch"
    config:
      model_name: "knowledgator/gliner-linker-large-v1.0"
```

### Production (Multi-Layer)
```yaml
# configs/pipelines/postgres_redis/default.yaml
name: "production_pipeline"
nodes:
  - id: "l2"
    processor: "l2_chain"
    config:
      layers:
        - type: "redis"
          priority: 2
          ttl: 3600
        - type: "postgres"
          priority: 0
```

## Advanced Features

### Precomputed Embeddings (BiEncoder)

**Save embeddings during entity loading:**
```python
executor.load_entities("entities.jsonl", target_layers=["postgres"])
executor.precompute_embeddings(target_layers=["postgres"], batch_size=64)
```

**Use in L3:**
```python
builder.l3.configure(
    model="knowledgator/gliner-linker-large-v1.0",
    use_precomputed_embeddings=True  # 10-100x faster
)
```

### On-the-Fly Embedding Caching

```python
builder.l3.configure(
    model="knowledgator/gliner-linker-large-v1.0",
    cache_embeddings=True  # Cache embeddings as they're computed
)
```

### Custom Pipelines

```python
# Custom L1 processing pipeline
l1_processor = processor_registry.get("l1_spacy")(
    config_dict={"model": "en_core_sci_sm"},
    pipeline=[
        ("extract_entities", {}),
        ("filter_by_length", {"min_length": 3}),
        ("deduplicate", {}),
        ("sort_by_position", {})
    ]
)
```

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Detailed architecture and API reference
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for AI assistants
- **[configs/pipelines/README.md](configs/pipelines/README.md)** - Configuration examples
- **[docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md)** - Database setup guide

## Testing

```bash
# Run all tests
pytest

# Run specific layer tests
pytest tests/l1/
pytest tests/l2/

# Run with coverage
pytest --cov=glinker --cov-report=html
```

## Citations

If you find GLiNKER useful in your research, please consider citing our papers:

```bibtex
@misc{stepanov2024glinermultitaskgeneralistlightweight,
      title={GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks}, 
      author={Ihor Stepanov and Mykhailo Shtopko},
      year={2024},
      eprint={2406.12925},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.12925}, 
}
```

## Contributing

We welcome contributions! Areas of interest:

- **Database layers** (MongoDB, Neo4j, vector databases)
- **Performance optimizations**
- **Documentation improvements**

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **GLiNER** - Zero-shot NER and entity linking ([urchade/GLiNER](https://github.com/urchade/GLiNER))
- **spaCy** - Industrial-strength NLP ([explosion/spaCy](https://github.com/explosion/spaCy))

## Contact

- **GitHub**: [Knowledgator/EntityLinker](https://github.com/Knowledgator/EntityLinker)
- **Email**: info@knowledgator.com

---

Developed by [Knowledgator](https://knowledgator.com)
