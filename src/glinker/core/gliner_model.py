"""GLinkerModel — BiEncoderTokenGLiNER subclass with sparse per-span scoring.

Architecture
------------
SparseTokenModel subclasses BiEncoderTokenModel and overrides forward().
The scorer itself is untouched — sparse packing/unpacking wraps around it:

    preprocess_sparse(token_rep, label_rep)
        → packs M mentions x their candidates into (M_total, max_sl, max_lc, H)

    self.scorer(super_tok, super_lbl)
        → (M_total, max_sl, max_lc, 3)  ← one GPU call vs O(seq x labels)

    postprocess_sparse(packed_scores, meta)
        → full (B, seq_len, num_classes, 3), fill=-10 for unscored positions

Without sparse config (_word_spans is None), forward is identical to parent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
from gliner.modeling.base import BiEncoderTokenModel
from gliner.modeling.outputs import GLiNERBaseOutput

try:
    from gliner.model import BiEncoderTokenGLiNER
except ImportError:
    BiEncoderTokenGLiNER = None

logger = logging.getLogger(__name__)

SPARSE_FILL: float = -10.0  # sigmoid(-10) ≈ 0 — below any practical threshold


# ---------------------------------------------------------------------------
# SparseTokenModel
# ---------------------------------------------------------------------------


def _pad_dim(t: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    """Pad tensor along *dim* to *target* length with zeros."""
    delta = target - t.shape[dim]
    if delta <= 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = delta
    return torch.cat([t, t.new_zeros(pad_shape)], dim=dim)


class SparseTokenModel(BiEncoderTokenModel):
    """BiEncoderTokenModel with built-in sparse per-span scoring.

    Two new methods handle the optimisation:

    ``preprocess_sparse(token_rep, label_rep)``
        For each batch item, gathers only the word positions that belong to a
        candidate mention and only the label embeddings that are candidates for
        that mention.  All items are packed into a single super-batch tensor so
        the scorer is called exactly once.

    ``postprocess_sparse(packed_scores, meta)``
        Scatters the packed scores back into a full (B, seq_len, num_classes, 3)
        output tensor.  Unscored positions are filled with SPARSE_FILL.

    When ``_word_spans`` is None (dense / fallback path), ``forward`` is
    byte-for-byte identical to the parent implementation — no overhead.
    """

    def __init__(self, config: Any, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._word_spans: List[List[Tuple[int, int]]] | None = None
        self._label_indices: List[List[List[int]]] | None = None

    # ------------------------------------------------------------------
    # Sparse config management
    # ------------------------------------------------------------------

    def set_sparse_config(
        self,
        word_spans: List[List[Tuple[int, int]]],
        label_indices: List[List[List[int]]],
    ) -> None:
        """Activate sparse mode for the next forward call.

        Args:
            word_spans: ``word_spans[b][m] = (word_start, word_end)`` inclusive.
            label_indices: ``label_indices[b][m]`` — indices into label_rep for mention m.
        """
        self._word_spans = word_spans
        self._label_indices = label_indices

    def clear_sparse_config(self) -> None:
        """Deactivate sparse mode (restores dense behaviour)."""
        self._word_spans = None
        self._label_indices = None

    # ------------------------------------------------------------------
    # Sparse pre/post processing
    # ------------------------------------------------------------------

    def preprocess_sparse(
        self,
        token_rep: torch.Tensor,  # (B, seq_len, H)
        label_rep: torch.Tensor,  # (B, num_classes, H)
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, dict]:
        """Pack sparse mentions into a single scorer call.

        Returns:
            super_tok:  ``(M_total, max_sl, H)``  — None if no valid spans.
            super_lbl:  ``(M_total, max_lc, H)``  — None if no valid spans.
            meta: scatter metadata consumed by ``postprocess_sparse``.
        """
        batch_size, seq_len, _H = token_rep.shape
        device = token_rep.device

        all_tok: List[torch.Tensor] = []
        all_lbl: List[torch.Tensor] = []
        batch_meta: List[dict | None] = []

        for b in range(batch_size):
            ws_list = self._word_spans[b] if b < len(self._word_spans) else []
            li_list = self._label_indices[b] if b < len(self._label_indices) else []

            if not ws_list or not li_list:
                batch_meta.append(None)
                continue

            M = len(ws_list)
            ws_t = torch.tensor([s for s, _ in ws_list], device=device)  # (M,)
            we_t = torch.tensor([e for _, e in ws_list], device=device)  # (M,)
            slen_t = we_t - ws_t + 1  # (M,)
            max_sl = int(slen_t.max())

            n_lbl = torch.tensor([len(li) for li in li_list], device=device)  # (M,)
            max_lc = int(n_lbl.max())
            if max_lc == 0:
                batch_meta.append(None)
                continue

            # ── gather token reps: (M, max_sl, H) ────────────────────────
            t_range = torch.arange(max_sl, device=device)
            tok_pos = (ws_t.unsqueeze(1) + t_range).clamp(0, seq_len - 1)  # (M, max_sl)
            tok = token_rep[b][tok_pos]  # (M, max_sl, H)
            tok_valid = t_range < slen_t.unsqueeze(1)  # (M, max_sl)
            tok = tok * tok_valid.unsqueeze(-1)  # zero padding

            # ── gather label reps: (M, max_lc, H) ────────────────────────
            lbl_idx = torch.zeros(M, max_lc, device=device, dtype=torch.long)
            for m, li in enumerate(li_list):
                if li:
                    lbl_idx[m, : len(li)] = torch.tensor(li, device=device)
            lbl = label_rep[b][lbl_idx]  # (M, max_lc, H)

            all_tok.append(tok)
            all_lbl.append(lbl)
            batch_meta.append(
                {
                    "b": b,
                    "M": M,
                    "tok_pos": tok_pos,  # (M, max_sl)
                    "lbl_idx": lbl_idx,  # (M, max_lc)
                    "tok_valid": tok_valid,  # (M, max_sl)
                    "n_lbl": n_lbl,  # (M,)
                }
            )

        meta = {
            "batch_meta": batch_meta,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_classes": label_rep.shape[1],
            "device": device,
            "dtype": token_rep.dtype,
        }

        if not all_tok:
            return None, None, meta

        # pad all items to the global (max_sl, max_lc)
        global_sl = max(t.shape[1] for t in all_tok)
        global_lc = max(l.shape[1] for l in all_lbl)

        super_tok = torch.cat([_pad_dim(t, global_sl, 1) for t in all_tok], dim=0)
        super_lbl = torch.cat([_pad_dim(l, global_lc, 1) for l in all_lbl], dim=0)
        return super_tok, super_lbl, meta

    def postprocess_sparse(
        self,
        packed_scores: torch.Tensor | None,  # (M_total, global_sl, global_lc, 3)
        meta: dict,
    ) -> torch.Tensor:
        """Scatter packed scores back into a full (B, seq_len, num_classes, 3) tensor.

        Unscored positions are pre-filled with SPARSE_FILL (≈ 0 after sigmoid).
        """
        output = torch.full(
            (meta["batch_size"], meta["seq_len"], meta["num_classes"], 3),
            fill_value=SPARSE_FILL,
            device=meta["device"],
            dtype=meta["dtype"],
        )
        if packed_scores is None:
            return output

        offset = 0
        for bm in meta["batch_meta"]:
            if bm is None:
                continue
            b, M = bm["b"], bm["M"]
            tok_pos = bm["tok_pos"]  # (M, local_sl)
            lbl_idx = bm["lbl_idx"]  # (M, local_lc)
            tok_valid = bm["tok_valid"]  # (M, local_sl)
            n_lbl = bm["n_lbl"]  # (M,)

            # slice back to local (potentially smaller) shapes
            scores_b = packed_scores[
                offset : offset + M,
                : tok_pos.shape[1],
                : lbl_idx.shape[1],
            ]  # (M, local_sl, local_lc, 3)

            l_range = torch.arange(lbl_idx.shape[1], device=meta["device"])
            lbl_valid = l_range < n_lbl.unsqueeze(1)  # (M, local_lc)
            valid = tok_valid.unsqueeze(2) & lbl_valid.unsqueeze(1)  # (M, local_sl, local_lc)
            vm, vt, vl = valid.nonzero(as_tuple=True)
            output[b, tok_pos[vm, vt], lbl_idx[vm, vl]] = scores_b[vm, vt, vl]

            offset += M

        return output

    # ------------------------------------------------------------------
    # Override forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_embeds=None,
        labels_input_ids=None,
        labels_attention_mask=None,
        words_embedding=None,
        mask=None,
        span_idx=None,
        span_mask=None,
        span_labels=None,
        prompts_embedding=None,
        prompts_embedding_mask=None,
        words_mask=None,
        text_lengths=None,
        labels=None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> GLiNERBaseOutput:
        encoder_kwargs = {
            key: kwargs[key]
            for key in ("packing_config", "pair_attention_mask", "token_lengths", "word_lengths")
            if key in kwargs
        }

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids,
            attention_mask,
            labels_embeds,
            labels_input_ids,
            labels_attention_mask,
            text_lengths,
            words_mask,
            **encoder_kwargs,
        )

        if labels is not None:
            words_embedding, mask = self._fit_length(words_embedding, mask, labels.shape[1])
            target_C = max(prompts_embedding.size(1), labels.size(-2))
            prompts_embedding, prompts_embedding_mask = self._fit_length(
                prompts_embedding, prompts_embedding_mask, target_C
            )

        # ── scoring: sparse or dense ───────────────────────────────────────
        if self._word_spans is not None:
            super_tok, super_lbl, meta = self.preprocess_sparse(words_embedding, prompts_embedding)
            packed = self.scorer(super_tok, super_lbl) if super_tok is not None else None
            scores = self.postprocess_sparse(packed, meta)
        else:
            scores = self.scorer(words_embedding, prompts_embedding)

        # ── optional span representations ─────────────────────────────────
        if getattr(self.config, "represent_spans", False):
            span_rep, span_idx, span_mask = self.get_span_representations(
                scores, span_idx, span_mask, words_embedding, labels, threshold
            )
            span_logits = torch.einsum("BND,BCD->BNC", span_rep, prompts_embedding)
        else:
            span_logits, span_idx, span_mask = None, None, None

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, mask, **kwargs)
            if span_labels is not None:
                span_loss = self.loss(
                    span_logits, span_labels, prompts_embedding_mask, span_mask, **kwargs
                )
                loss = self.config.token_loss_coef * loss + self.config.span_loss_coef * span_loss

        return GLiNERBaseOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
            span_idx=span_idx,
            span_logits=span_logits,
            span_mask=span_mask,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_word_spans(
    gliner: Any,
    text: str,
    char_spans: List[Dict[str, int]],
    span_label_indices: List[List[int]],
) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """Convert char-level spans to word-level spans.

    Uses GLiNER's own words_splitter so boundaries match exactly.
    Spans that don't align to word boundaries are silently dropped.

    Args:
        gliner: GLiNER instance (provides ``prepare_inputs``).
        text: Input text.
        char_spans: ``[{"start": int, "end": int}, ...]`` from L1.
        span_label_indices: Per-span candidate indices aligned with char_spans.

    Returns:
        ``(word_spans, label_indices)`` — only successfully matched spans.
    """
    _, start_maps, end_maps = gliner.prepare_inputs([text])
    s2w = {char: word for word, char in enumerate(start_maps[0])}
    e2w = {char: word for word, char in enumerate(end_maps[0])}

    word_spans: List[Tuple[int, int]] = []
    matched_lbl: List[List[int]] = []

    for span, lbl_idxs in zip(char_spans, span_label_indices):
        ws = s2w.get(span["start"])
        we = e2w.get(span["end"])
        if ws is not None and we is not None and we >= ws:
            word_spans.append((ws, we))
            matched_lbl.append(list(lbl_idxs))

    return word_spans, matched_lbl


# ---------------------------------------------------------------------------
# GLinkerModel
# ---------------------------------------------------------------------------


class GLinkerModel(BiEncoderTokenGLiNER):
    """BiEncoderTokenGLiNER with sparse per-span scoring.

    Replaces the inner model with SparseTokenModel so that scoring is done only
    over per-mention candidate labels when ``span_label_indices`` is provided.

    Usage::

        model = GLinkerModel.from_pretrained("knowledgator/gliner-linker-large-v1.0")

        entities = model.predict_entities(
            text,
            labels,                           # flat union of all candidates
            span_label_indices=[[0,1],[2,3]], # per-mention candidate indices
            input_spans=[{"start":0,"end":5}, {"start":10,"end":18}],
        )
    """

    # ← This is the only line that replaces SparseScorer installation:
    model_class = SparseTokenModel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_entities(
        self,
        text: str,
        labels: List[str],
        span_label_indices: List[List[int]] | None = None,
        input_spans: List[Dict[str, int]] | None = None,
        flat_ner: bool = True,
        threshold: float = 0.5,
        span_extraction_threshold: float | None = None,
        multi_label: bool = False,
        return_class_probs: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if span_label_indices is None:
            return super().predict_entities(
                text,
                labels,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                return_class_probs=return_class_probs,
                **kwargs,
            )

        word_spans, word_lbl_indices = build_word_spans(self, text, input_spans, span_label_indices)
        return self.inference(
            [text],
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
            input_spans=[input_spans],
            _word_spans=[word_spans],
            _label_indices=[word_lbl_indices],
            _span_extraction_threshold=span_extraction_threshold,
            **kwargs,
        )[0]

    def predict_with_embeds(
        self,
        text: str,
        labels_embeddings: torch.Tensor,
        labels: List[str],
        span_label_indices: List[List[int]] | None = None,
        input_spans: List[Dict[str, int]] | None = None,
        flat_ner: bool = True,
        threshold: float = 0.5,
        span_extraction_threshold: float | None = None,
        multi_label: bool = False,
        return_class_probs: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        lbl_emb = labels_embeddings.to(self.device)

        if span_label_indices is None:
            return self.inference(
                [text],
                labels,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                return_class_probs=return_class_probs,
                labels_embeds=lbl_emb,
                _span_extraction_threshold=span_extraction_threshold,
                **kwargs,
            )[0]

        word_spans, word_lbl_indices = build_word_spans(self, text, input_spans, span_label_indices)
        return self.inference(
            [text],
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
            input_spans=[input_spans],
            labels_embeds=lbl_emb,
            _word_spans=[word_spans],
            _label_indices=[word_lbl_indices],
            _span_extraction_threshold=span_extraction_threshold,
            **kwargs,
        )[0]

    # ------------------------------------------------------------------
    # Batch processing — injects sparse config per batch
    # ------------------------------------------------------------------

    def _process_batches(
        self,
        data_loader,
        threshold,
        flat_ner,
        multi_label,
        packing_config=None,
        return_class_probs=False,
        word_input_spans=None,
        _word_spans: List | None = None,
        _label_indices: List | None = None,
        _span_extraction_threshold: float | None = None,
        **external_inputs,
    ):
        """Override _process_batches to handle sparse scoring and external inputs.

        Delegates to the parent ONLY when there is nothing special to do (no sparse
        config, no external inputs like labels_embeds).  Otherwise uses its own loop
        because the installed GLiNER has a bug: it ignores ``external_inputs`` when
        ``packing_config is None`` (merges only when packing_config is set).

        ``_span_extraction_threshold`` is passed to ``model.forward`` (controls BIO
        span extraction), while ``threshold`` is used by ``decoder.decode`` (filters
        final span_logits).  When None, ``threshold`` is used for both.
        """
        if _word_spans is None and not external_inputs and _span_extraction_threshold is None:
            return super()._process_batches(
                data_loader,
                threshold,
                flat_ner,
                multi_label,
                packing_config=packing_config,
                return_class_probs=return_class_probs,
                word_input_spans=word_input_spans,
            )

        # BIO extraction uses a separate (typically lower) threshold if provided.
        bio_threshold = (
            _span_extraction_threshold if _span_extraction_threshold is not None else threshold
        )

        # Custom loop: handles sparse config AND external inputs (e.g. labels_embeds).
        outputs = []
        device = self.device
        batch_offset = 0

        for batch in data_loader:
            if not getattr(self, "onnx_model", False):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

            current_batch_size = len(batch["tokens"])
            model_inputs = {**batch, **external_inputs}
            if packing_config is not None:
                model_inputs["packing_config"] = packing_config

            if _word_spans is not None:
                batch_ws = _word_spans[batch_offset : batch_offset + current_batch_size]
                batch_li = _label_indices[batch_offset : batch_offset + current_batch_size]
                self.model.set_sparse_config(batch_ws, batch_li)

            try:
                with torch.inference_mode():
                    model_output = self.model(**model_inputs, threshold=bio_threshold)
            finally:
                if _word_spans is not None:
                    self.model.clear_sparse_config()

            model_logits = model_output[0]
            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)

            batch_input_spans = None
            if word_input_spans is not None:
                batch_input_spans = word_input_spans[
                    batch_offset : batch_offset + current_batch_size
                ]

            decoded = self.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_logits,
                span_idx=model_output.span_idx,
                span_mask=model_output.span_mask,
                span_logits=model_output.span_logits,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                return_class_probs=return_class_probs,
                input_spans=batch_input_spans,
            )
            outputs.extend(decoded)
            batch_offset += current_batch_size

        return outputs
