import sys
sys.path.append('.')

import src.l0
import src.l1
import src.l2
import src.l3

from src.core.pipeline import Pipeline
from src.l1.models import L1Input

texts = [
    "TP53 is a tumor suppressor gene that activates p21 expression.",
    "Cyclin D1 (CCND1) is overexpressed in many cancers and promotes cell proliferation.",
    "The EMT process is regulated by p53 signaling pathways in cancer cells."
]

print("="*70)
print("PIPELINE ORCHESTRATOR TEST")
print("="*70)

# Load pipeline
pipeline = Pipeline.from_yaml("configs/pipelines/default.yaml", verbose=True)

print(f"\nProcessing {len(texts)} texts\n")

# Create L1Input and run pipeline
l1_input = L1Input(texts=texts)
result = pipeline(l1_input)

# Get outputs
l3_output = result.get('l3_output')

# Display
print("\n" + "="*70)
print("RESULTS")
print("="*70)

for i, (text, entities) in enumerate(zip(texts, l3_output.entities), 1):
    print(f"\nText {i}: {len(entities)} entities")
    print(f"\"{text}\"\n")
    
    if entities:
        for j, e in enumerate(entities, 1):
            print(f"  {j}. '{e.text}' → {e.label}")
            print(f"     Score: {e.score:.3f} | Position: {e.start}-{e.end}")
    else:
        print("  (No entities found)")

l2_input = result.get('l2_input')
l3_input = result.get('l3_input')

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Texts:       {len(texts)}")
print(f"Mentions:    {len(l2_input.mentions)}")
print(f"Candidates:  {sum(len(c) for c in l3_input.candidates)}")
print(f"Entities:    {sum(len(e) for e in l3_output.entities)}")
print("="*70)