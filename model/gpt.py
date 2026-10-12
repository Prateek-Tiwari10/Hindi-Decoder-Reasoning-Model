"""The decoder-only Transformer, built from ``nn.Linear``, ``nn.Embedding``
and ``nn.LayerNorm`` only.

The brief forbids ``nn.Transformer*``, HuggingFace model classes and any
pre-built attention (including ``F.scaled_dot_product_attention``).  The
positional embeddings, the multi-head split, the causal mask and the
attention arithmetic below are therefore written out by hand.

SHAPE GLOSSARY -- used consistently throughout this file
--------------------------------------------------------
    B   batch size (sequences per forward pass)
    T   time / sequence length, at most context_len = 512
    C   channels, ``d_model`` = 384
    H   heads = 6
    Dk  per-head channels, ``d_k`` = C // H = 64
    V   vocabulary = 16,000

THE SHAPE OF ONE FORWARD PASS
-----------------------------
    ids            (B, T)              int64 token ids
    -> embeddings  (B, T, C)           token embedding + position embedding
    -> x11 blocks  (B, T, C)           shape is invariant through the stack
    -> final norm  (B, T, C)
    -> logits      (B, T, V)           tied to the input embedding

The stack preserves ``(B, T, C)`` exactly.  That invariance is why depth is a
free parameter: the blocks compose because each one is an endomorphism on the
residual stream.

WHY THIS SHAPE OF MODEL (narrow-deep)
-------------------------------------
The 25-30M budget is spent as 11 layers x 384 channels.  The token embedding
costs ``V * C`` = 6.14M parameters -- 24% of the budget -- and that cost is
linear in ``C`` while the per-layer cost is quadratic in ``C``.  Keeping ``C``
small buys depth: it shrinks the embedding tax and makes each layer cheap.

This is the GPT-style baseline of the architecture study: learned absolute
positions, pre-norm LayerNorm, a GELU MLP with biases, tied embeddings.  Its
behaviour is frozen -- ``artifacts/checkpoints/gpt/final_16k.pt`` must load
into it with ``strict=True`` and reproduce its recorded test loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import causal_mask, count_parameters
from model.config import ModelConfig


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out by hand.

    WHAT IT COMPUTES
        For every position ``t``, a weighted average of the *values* at
        positions ``0..t``, where the weights come from how well the query at
        ``t`` matches the key at each of those positions.  This is the only
        module in the network where positions exchange information -- the
        feed-forward and the norms act on each position independently.  Remove
        it and the model collapses to a per-token bigram-ish classifier that
        cannot use context at all.

    IN / OUT
        ``(B, T, C)`` -> ``(B, T, C)``, plus optionally the attention weights
        ``(B, H, T, T)``.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the projections and the causal mask.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_k

        # Three separate projections rather than one fused (C, 3C) matrix.
        # Mathematically identical and slightly slower, but it keeps Q, K and V
        # nameable as distinct objects, which is the point of writing this by
        # hand.
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # Mixes the heads' outputs back together.  Without it each head's
        # contribution would stay confined to its own 64-channel slice of the
        # residual stream and the heads could never combine.
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # --- the causal mask ---
        # An ADDITIVE mask of shape (1, 1, T, T): 0.0 where attention is
        # allowed, -inf strictly above the diagonal (the future).  Adding -inf
        # to a score sends exp(score) to exactly 0 in the softmax, so a future
        # position receives *zero* weight -- not merely a small one.
        #
        # Additive-before-softmax, not multiplicative-after: zeroing weights
        # after the softmax would leave the remaining weights summing to less
        # than 1, silently scaling down the output at early positions.
        #
        # Row t of the mask allows columns 0..t, so position t can always see
        # itself.  No row is ever fully masked, which is what keeps softmax
        # from producing NaN.
        #
        # Registered non-persistently: it is a constant derived from
        # context_len, so writing it into every checkpoint would be dead weight.
        self.register_buffer("causal_mask", causal_mask(config.context_len),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run causal self-attention.

        Args:
            x: Input activations, ``(B, T, C)``.
            return_attention: If True, also return the post-softmax attention
                weights.  The attention analysis's heatmaps, entropy and distance statistics
                all read them from here.

        Returns:
            ``(output, attention)`` where output is ``(B, T, C)`` and attention
            is ``(B, H, T, T)`` or None.
        """
        B, T, C = x.shape

        # --- project to queries, keys, values: (B, T, C) -> (B, T, C) each ---
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --- split the channel axis into heads ---
        # (B, T, C) -view-> (B, T, H, Dk) -transpose-> (B, H, T, Dk)
        #
        # The view is a pure reinterpretation of the last axis: channel j goes
        # to head j // Dk, slot j % Dk.  The transpose then puts H next to B so
        # the matmuls below treat (B, H) as a stack of independent problems --
        # every head attends separately, which is the whole idea.
        #
        # The inverse is exactly transpose-then-reshape, applied after the
        # weighted sum; see the merge step below.
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # --- scaled dot-product scores ---
        # (B, H, T, Dk) @ (B, H, Dk, T) -> (B, H, T, T)
        # scores[b, h, i, j] = <query at i, key at j> for head h.
        #
        # WHY 1/sqrt(d_k): the dot product of two Dk-dimensional vectors with
        # unit-variance independent entries has variance Dk, so raw scores grow
        # like sqrt(Dk) = sqrt(56) ~ 7.5.  Feed scores that large into softmax
        # and it saturates: one weight goes to ~1, the rest to ~0, and the
        # gradient through the softmax vanishes.  Dividing by sqrt(d_k)
        # renormalises the scores to unit variance so training starts in the
        # softmax's responsive region.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # --- mask the future, BEFORE the softmax ---
        # Sliced to (1, 1, T, T) so shorter sequences work; broadcasts over B, H.
        # This is what makes the model a *language* model: the prediction at
        # position t must not depend on tokens after t, or the training
        # objective would be trivially solvable by copying the answer.
        scores = scores + self.causal_mask[:, :, :T, :T]

        # --- softmax over the KEY axis ---
        # dim=-1 is j, the position being attended TO.  Each query's weights
        # then sum to 1 across the positions it may see.  Softmaxing over the
        # query axis instead would be meaningless -- queries do not compete.
        attention = F.softmax(scores, dim=-1)

        # Dropout is applied to a COPY used for the output, so the weights we
        # hand back for analysis are the true post-softmax distribution and
        # still sum to 1.  (In eval mode dropout is identity and the two agree.)
        weights = self.attn_dropout(attention)

        # --- weighted sum of values ---
        # (B, H, T, T) @ (B, H, T, Dk) -> (B, H, T, Dk)
        out = weights @ v

        # --- merge the heads: the exact inverse of the split above ---
        # (B, H, T, Dk) -transpose-> (B, T, H, Dk) -reshape-> (B, T, C)
        # contiguous() is required because transpose only permutes strides;
        # reshape needs the memory laid out in the new order.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.out_proj(out))
        return out, (attention if return_attention else None)


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    WHAT IT COMPUTES
        The same two-layer MLP applied independently at every position:
        ``C -> 4C -> C``.  Attention moves information *between* positions;
        this is where the model does nonlinear computation *at* a position.
        Without it the whole network would be a stack of linear maps
        interleaved with averaging, and depth would buy nothing.

    WHY 4x
        The 4x expansion is the standard ratio and is where most of the
        parameters live: 1.18M of each block's 1.77M.  The widening gives the
        nonlinearity room to represent features that the 384-channel residual
        stream is too narrow to hold directly.

    IN / OUT
        ``(B, T, C)`` -> ``(B, T, C)``.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the two projections.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        GELU rather than ReLU: it is smooth everywhere, so unlike ReLU it has
        no dead region where the gradient is identically zero, and it is what
        the GPT family uses.

        Args:
            x: ``(B, T, C)``.

        Returns:
            ``(B, T, C)``.
        """
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """One pre-norm decoder block.

    PRE-NORM, NOT POST-NORM
        Each sublayer computes ``x = x + sublayer(norm(x))``.  The alternative,
        post-norm (``x = norm(x + sublayer(x))``), puts a LayerNorm *on* the
        residual path, so the identity shortcut from the loss back to the
        embeddings is interrupted at every one of the 14 sublayers.  With
        pre-norm the residual path is a clean sum: gradients reach layer 0
        undistorted, and the model trains without a warmup-and-pray schedule.
        This is why modern decoders are pre-norm, and why the plan locks it.

        The cost is that the residual stream is never normalised on its own, so
        its magnitude grows with depth -- which is exactly what the final
        LayerNorm in :class:`DecoderLM` is there to fix.

    IN / OUT
        ``(B, T, C)`` -> ``(B, T, C)``.  Shape-preserving, which is what lets
        the blocks stack.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the two sublayers and their norms.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.norm_ffn = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run both sublayers.

        Args:
            x: ``(B, T, C)``.
            return_attention: Forwarded to the attention sublayer.

        Returns:
            ``(output, attention)``; output is ``(B, T, C)``.
        """
        attn_out, attention = self.attn(self.norm_attn(x), return_attention)
        x = x + attn_out                       # residual add, not overwrite
        x = x + self.ffn(self.norm_ffn(x))
        return x, attention


class DecoderLM(nn.Module):
    """The full decoder-only language model.

    Attributes:
        config: The architecture this instance was built from.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build embeddings, the block stack and the tied output head.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # LEARNED ABSOLUTE POSITIONS.  A lookup table with one row per position.
        # Attention is a weighted sum and is therefore permutation-equivariant:
        # with no positional signal, "raam ne shyaam ko dekha" and
        # "shyaam ne raam ko dekha" would produce identical representations.
        # This table is the only thing in the model that knows about order.
        #
        # Its height IS the hard maximum sequence length.  Position 512 has no
        # row, so a longer sequence raises an IndexError rather than silently
        # doing something wrong -- unlike sinusoidal or rotary encodings, which
        # extrapolate (badly) instead of failing loudly.
        #
        # THE ABLATION.  With positional="none" the table is not created at
        # all.  Attention is a weighted sum and therefore permutation-
        # equivariant, so without it the model is a bag of tokens: it can still
        # learn which tokens co-occur, but not in what order.  Setting the
        # table to zeros would leave dead parameters in the optimiser and in
        # every checkpoint; omitting it makes the absence structural.
        if config.positional == "none":
            self.position_embedding = None
        else:
            self.position_embedding = nn.Embedding(config.context_len,
                                                   config.d_model)

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers))

        # Pre-norm leaves the residual stream unnormalised, and after 11 blocks
        # of additions its scale is large and depth-dependent.  This final norm
        # puts it back on a fixed scale before the output head, which the tied
        # head in particular depends on: the head's weights are the *embedding*
        # matrix, which was never trained to consume a large-magnitude input.
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # bias=False because the tied head must be exactly the transpose of the
        # embedding matrix; a bias would be an untied extra.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            # WEIGHT TYING.  One (V, C) matrix serves both jobs: read a row to
            # embed a token, and multiply by the whole matrix to score every
            # token.  This is an assignment of the same tensor object, not a
            # copy -- one set of gradients, one set of optimiser state.
            #
            # It saves V*C = 6,144,000 parameters, 19% of what the model would
            # otherwise cost -- about three and a half blocks' worth on a 25.9M
            # budget.  The justification is not only thrift: the
            # embedding and the output head are inverse operations over the same
            # vocabulary, so tying imposes a true symmetry and acts as a
            # regulariser, which matters most for the rare tokens that a 500M
            # corpus sees only a handful of times.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # GPT-2's scaled init: the two projections that write INTO the residual
        # stream get their std divided by sqrt(2 * n_layers).  Each block adds
        # two contributions, so without this the stream's variance grows
        # linearly with depth and the late blocks start out shouting over the
        # early ones.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc_out.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise one module, GPT-2 style.

        std=0.02 is small enough that the initial logits are near-uniform,
        which is what makes the initial loss land on ``ln(V)``.

        Args:
            module: Module to initialise in place.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count real parameter tensors.

        Uses the ``id`` of each tensor so the tied matrix is counted once --
        ``parameters()`` already de-duplicates shared tensors, and this makes
        that reliance explicit rather than incidental.

        Args:
            trainable_only: Count only parameters requiring gradients.

        Returns:
            Number of scalars.
        """
        return count_parameters(self, trainable_only)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attention: bool = False,
                last_position_only: bool = False):
        """Run the model.

        Args:
            idx: Token ids, ``(B, T)``, int64, every entry below ``vocab_size``
                and ``T <= context_len``.
            targets: Next-token ids, ``(B, T)``.  If given, the loss is
                returned too.
            return_attention: Collect per-layer attention weights.
            last_position_only: Compute logits for the final position only.
                Autoregressive generation appends one token at a time and uses
                nothing but the last row, yet the output head is the single
                most expensive layer -- projecting to 24,000 classes at *every*
                position. Skipping the rest turns a ``(B, T, 24000)`` tensor
                into ``(B, 1, 24000)``, which at B=25, T=192 is 460 MB versus
                2 MB. Ignored when ``targets`` is given, since the loss needs
                every position.

        Returns:
            ``(logits, loss, attentions)``.  ``logits`` is ``(B, T, V)``,
            ``loss`` is a scalar or None, ``attentions`` is a list of
            ``n_layers`` tensors of shape ``(B, H, T, T)`` or None.

        Raises:
            ValueError: If the sequence is longer than the positional table.
        """
        B, T = idx.shape
        if T > self.config.context_len:
            raise ValueError(
                f"sequence length {T} exceeds context_len "
                f"{self.config.context_len}; the positional table has no row "
                f"for position {T - 1}")

        # (B, T) -> (B, T, C)
        token_emb = self.token_embedding(idx)

        # (T,) -> (T, C), broadcast over the batch: every sequence in the batch
        # uses the same positions, so one row vector per position suffices.
        if self.position_embedding is None:
            # The ablation: nothing is added, so the residual stream carries
            # "what" with no "where" at all.
            x = self.embedding_dropout(token_emb)
        else:
            positions = torch.arange(T, device=idx.device)
            pos_emb = self.position_embedding(positions)

            # Summed, not concatenated: concatenating would spend channels on
            # position that could otherwise carry content, and would change C.
            # The residual stream carries "what" and "where" superposed, and
            # the first attention layer learns to separate them.
            x = self.embedding_dropout(token_emb + pos_emb)

        attentions = [] if return_attention else None
        for block in self.blocks:
            x, attention = block(x, return_attention)
            if return_attention:
                attentions.append(attention)

        x = self.final_norm(x)
        if last_position_only and targets is None:
            # Keepdim so the result stays rank-3 and callers can index [:, -1]
            # uniformly whether or not this shortcut was taken.
            x = x[:, -1:, :]
        logits = self.lm_head(x)                       # (B, T or 1, V)

        loss = None
        if targets is not None:
            # Cross-entropy wants (N, V) against (N,), so both are flattened
            # over batch and time: every position in every sequence is one
            # independent prediction, and the loss is their mean.
            #
            # The alignment -- logits at t against the token at t+1 -- was
            # already done in the data pipeline, where y is x shifted by one.
            # Shifting again here would double-shift and teach the model to
            # predict two tokens ahead.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, attentions


def build_model(config: ModelConfig, device: str = "cpu",
                verify_budget: bool = True) -> DecoderLM:
    """Construct a model and check it against its design.

    Args:
        config: Architecture configuration.
        device: Where to place the model.
        verify_budget: Assert the live parameter count matches the analytic
            one and sits inside the brief's 25-30M window.

    Returns:
        The model, on ``device``.

    Raises:
        AssertionError: If the built model disagrees with the designed one.
    """
    model = DecoderLM(config).to(device)
    if verify_budget:
        actual, expected = model.num_parameters(), config.parameter_count
        assert actual == expected, (
            f"built model has {actual:,} parameters but the config predicts "
            f"{expected:,}; the code and the design have diverged")
        assert config.within_budget, (
            f"{actual:,} parameters is outside the 25-30M budget")
    return model
