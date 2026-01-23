# demo.py
import yaml
import os
import argparse

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gradio as gr
from src.core.factory import ProcessorFactory
import torch

# ========== CONFIGURATION ==========

# Default config - on-the-fly caching
DEFAULT_CONFIG = "configs/pipelines/demo_onthefly_cache.yaml"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Entity Linking Pipeline Demo")
parser.add_argument(
    "--config", "-c",
    type=str,
    default=DEFAULT_CONFIG,
    help=f"Path to pipeline config YAML (default: {DEFAULT_CONFIG})"
)
parser.add_argument(
    "--entities", "-e",
    type=str,
    default="pubmesh_ontology.jsonl",
    help="Path to entities JSONL file (default: pubmesh_ontology.jsonl)"
)
args, _ = parser.parse_known_args()

# ========== INITIALIZATION ==========

print(f"📄 Loading config from: {args.config}")
with open(args.config, 'r') as f:
    yaml_config = yaml.safe_load(f)

print("🚀 Initializing executor...")
executor = ProcessorFactory.create_from_dict(yaml_config, verbose=False)

print("📥 Loading entities...")
executor.load_entities(args.entities, target_layers=['dict'])

# Check if cache_embeddings is enabled in L3 config
l3_node = next((n for n in yaml_config.get('nodes', []) if n.get('id') == 'l3'), None)
cache_embeddings = l3_node.get('config', {}).get('cache_embeddings', False) if l3_node else False

if cache_embeddings:
    print("🔗 Setting up on-the-fly embedding caching...")
    executor.setup_l3_cache_writeback()
    cache_msg = "(Embeddings will be cached on first use)"
else:
    cache_msg = "(No embedding caching)"

print("📄 Loading example texts...")
texts = open('pubmed_texts.txt').readlines()
import random
random.seed(42)
random.shuffle(texts)
example_texts = texts[:10]

print(f"✅ Ready! {cache_msg}")

# ========== PROCESSING FUNCTION ==========

MAX_MENTIONS = 25
MAX_CANDIDATES = 50

def process_text(text: str, threshold: float):
    """Process single text through the pipeline"""

    try:
        if not text.strip():
            return {"text": "", "entities": []}, "❌ Please enter some text", "", "", ""

        # Execute pipeline
        input_data = {"texts": [text]}
        context = executor.execute(input_data)

        # Get L0 aggregated results
        l0_result = context.get("l0_result")

        if not l0_result or not l0_result.entities:
            return {"text": "", "entities": []}, "❌ No results", "", "", ""

        l0_entities = l0_result.entities[0] if l0_result.entities else []

        # ===== L0 OUTPUT - AGGREGATED VIEW =====
        linked_entities = [e for e in l0_entities if e.is_linked and e.linked_entity.confidence >= threshold]
        total_mentions = len(l0_entities)

        print(f"🎯 L0 aggregated: {total_mentions} mentions, {len(linked_entities)} linked")

        # Highlighted output
        highlighted_output = {
            "text": text,
            "entities": [
                {
                    "entity": entity.linked_entity.label[:75],
                    "word": entity.mention_text,
                    "start": entity.mention_start,
                    "end": entity.mention_end,
                    "score": round(entity.linked_entity.confidence, 3),
                }
                for entity in linked_entities
            ]
        }

        # L0 detailed output
        l0_output = f"**📊 Pipeline Statistics:**\n\n"
        l0_output += f"- Total mentions: {l0_result.stats['total_mentions']}\n"
        l0_output += f"- Linked: {l0_result.stats['linked']}\n"
        l0_output += f"- Linking rate: {l0_result.stats['linking_rate']:.1%}\n\n"
        l0_output += f"**🔗 Linked entities (threshold={threshold}):**\n\n"

        for i, ent in enumerate(linked_entities, 1):
            l0_output += f"**{i}. {ent.mention_text}** → {ent.linked_entity.label}\n"
            l0_output += f"   - **Entity ID:** {ent.linked_entity.entity_id}\n"
            l0_output += f"   - **Confidence:** {ent.linked_entity.confidence:.3f}\n"
            l0_output += f"   - **Position:** {ent.mention_start}-{ent.mention_end}\n"
            l0_output += f"   - **Candidates found:** {ent.num_candidates}\n\n"

        # ===== L1 OUTPUT - FROM L0 =====
        total_l1 = len(l0_entities)
        l1_entities_display = l0_entities[:MAX_MENTIONS]

        l1_output = f"**Found {total_l1} mentions (showing first {len(l1_entities_display)}):**\n\n"
        for i, ent in enumerate(l1_entities_display, 1):
            l1_output += f"{i}. **{ent.mention_text}** (pos: {ent.mention_start}-{ent.mention_end})\n"
            l1_output += f"   - Left context: ...{ent.left_context[-40:]}\n"
            l1_output += f"   - Right context: {ent.right_context[:40]}...\n\n"

        if total_l1 > MAX_MENTIONS:
            l1_output += f"\n⚠️ *...and {total_l1 - MAX_MENTIONS} more mentions not shown*\n"

        # ===== L2 OUTPUT - FROM L0 =====
        all_candidates = []
        for ent in l0_entities:
            all_candidates.extend(ent.candidates)

        # Deduplicate by entity_id
        seen_ids = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand.entity_id not in seen_ids:
                unique_candidates.append(cand)
                seen_ids.add(cand.entity_id)

        total_l2 = len(unique_candidates)
        l2_candidates = unique_candidates[:MAX_CANDIDATES]

        l2_output = f"**Found {total_l2} unique candidates (showing first {len(l2_candidates)}):**\n\n"
        for i, cand in enumerate(l2_candidates, 1):
            l2_output += f"**{i}. {cand.label}**\n"
            l2_output += f"   - ID: `{cand.entity_id}`\n"
            description = cand.description[:150] if cand.description else "N/A"
            l2_output += f"   - Description: {description}...\n"
            if cand.aliases:
                l2_output += f"   - Aliases: {', '.join(cand.aliases[:3])}\n"
            l2_output += "\n"

        if total_l2 > MAX_CANDIDATES:
            l2_output += f"\n⚠️ *...and {total_l2 - MAX_CANDIDATES} more candidates not shown*\n"

        # ===== L3 OUTPUT - FROM L0 =====
        l3_output = f"**Linked {len(linked_entities)} entities (threshold={threshold}):**\n\n"
        for i, ent in enumerate(linked_entities, 1):
            l3_output += f"**{i}. {ent.mention_text}** → {ent.linked_entity.label}\n"
            l3_output += f"   - **Score:** {ent.linked_entity.confidence:.3f}\n"
            l3_output += f"   - **Position:** {ent.mention_start}-{ent.mention_end}\n\n"

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return highlighted_output, l0_output, l1_output, l2_output, l3_output

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        print(error_msg)
        return {"text": "", "entities": []}, error_msg, error_msg, error_msg, error_msg


# ========== GRADIO INTERFACE ==========

config_name = yaml_config.get('name', 'unknown')
config_desc = yaml_config.get('description', '')

with gr.Blocks(title="Entity Linking Pipeline Demo") as demo:
    gr.Markdown(
        f"""
        # 🔗 Entity Linking Pipeline Demo

        **Config:** `{config_name}` - {config_desc}

        **4-Layer Entity Linking Pipeline:**
        - **L1**: spaCy NER extracts mentions
        - **L2**: Database search finds candidates (with embedding storage)
        - **L3**: GLiNER links mentions to entities
        - **L0**: Final aggregation with statistics

        {cache_msg}
        """
    )

    input_text = gr.Textbox(
        value=example_texts[0],
        label="Text input",
        placeholder="Enter your text here",
        lines=10
    )

    with gr.Row():
        threshold = gr.Slider(
            0,
            1,
            value=0.2,
            step=0.05,
            label="Threshold",
            info="Lower the threshold to increase how many entities get predicted.",
            scale=2
        )
        submit_btn = gr.Button("Submit", variant="primary", scale=1)

    output = gr.HighlightedText(label="Linked Entities")

    # ACCORDIONS
    with gr.Accordion("🎯 L0: Final Aggregation & Statistics", open=True):
        l0_output = gr.Markdown()

    with gr.Accordion("📊 L1: Named Entity Recognition (max 25 shown)", open=False):
        l1_output = gr.Markdown()

    with gr.Accordion("🔍 L2: Candidate Generation (max 50 shown)", open=False):
        l2_output = gr.Markdown()

    with gr.Accordion("🔗 L3: Entity Linking Details", open=False):
        l3_output = gr.Markdown()

    # Examples
    gr.Examples(
        examples=[[text, 0.2] for text in example_texts],
        inputs=[input_text, threshold],
        label="Example Texts from PubMed"
    )

    # Event handlers
    all_outputs = [output, l0_output, l1_output, l2_output, l3_output]

    submit_btn.click(
        fn=process_text,
        inputs=[input_text, threshold],
        outputs=all_outputs,
        show_progress="full"
    )
    
    # threshold.change(
    #     fn=process_text,
    #     inputs=[input_text, threshold],
    #     outputs=all_outputs,
    #     show_progress="full"
    # )

# ========== LAUNCH ==========

if __name__ == "__main__":
    # demo.queue(
    #     max_size=20,
    #     default_concurrency_limit=None
    # )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True,
        show_error=True
    )