# Clinical Database Demo

This example demonstrates **entity extraction and deduplication** for clinical databases using GLinker.

## 📋 Use Case

Healthcare systems need to extract structured data from unstructured clinical notes while avoiding duplicate database entries. This demo shows how GLinker can:

1. **Extract entities** (Patients, Doctors, Diseases) from clinical text using zero-shot learning
2. **Match name variations** (e.g., "Jon Doe" → "John Doe") through database aliases and L2/L3 matching
3. **Link to existing records** to prevent duplicate insertions
4. **Identify new entities** that need to be added to the database

## 🚀 Quick Start

Open and run [`clinical_db_demo.ipynb`](./clinical_db_demo.ipynb) for a complete walkthrough with explanations and outputs.

### Pipeline Layers

This demo implements the following GLinker layers:

| Layer | Purpose | Processor |
|-------|---------|-----------|
| **L1** | Mention extraction using GLiNER zero-shot NER | `l1_gliner` |
| **L2** | Candidate retrieval from dictionary database | `l2_chain` |
| **L3** | Entity disambiguation via GLiNER | `l3_batch` |
| **L0** | Aggregation, filtering, and final output | `l0_aggregator` |

### Models

- **L1 NER**: `knowledgator/gliner-bi-base-v2.0` (zero-shot entity extraction)
- **L3 Linking**: `knowledgator/gliner-bi-edge-v2.0` (entity disambiguation)

### Mock Database

The demo uses [`../data/mock_db.jsonl`](../data/mock_db.jsonl) containing:

- **Patients**: John Doe (P001), Sarah Connor (P002)
- **Doctors**: Dr. Gregory House (D001), Dr. Stephen Strange (D002)
- **Diseases**: Diabetes Mellitus (C001), Hypertension (C002)

Each record includes aliases for name variation handling (e.g., "Jon Doe", "John H Doe" for P001).

## 🎯 Key Features Demonstrated

### 1. Zero-Shot Learning
Extract custom entity types without training data using GLiNER with labels: `["Patient", "Doctor", "Disease", "Symptom"]`

### 2. Name Variation Handling
- "Jon Doe" → "John Doe" ✓ (via alias in database)
- "John H Doe" → "John Doe" ✓ (via alias in database)
- "Dr. House" → "Dr. Gregory House" ✓ (via alias in database)
- "high blood pressure" → "Hypertension" ✓ (via L2 fuzzy + L3 linking)

### 3. Entity Linking
Match extracted entities to database records with confidence scores.

### 4. Deduplication Logic
- **Existing entities**: Skip insertion, return database ID
- **New entities**: Flag for insertion into appropriate table

## 📝 Requirements
- Python 3.10+
- GLinker (with enhancements)
- Dependencies: Glinker