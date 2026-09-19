"""
KARMA-T: Knowledge Anchors for Robust Memory-Augmented Temporal Representation Learning
         with Transductive-Inductive Hybrid Encoder
model.py - Blends learned entity embeddings with inductive encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNeighborhoodEncoder(nn.Module):
    def __init__(self, embedding_dim, num_anchors):
        super().__init__()
        self.d = embedding_dim
        self.K = num_anchors

        self.rel_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.time_proj = nn.Linear(1, embedding_dim)
        self.nbr_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.attn_q    = nn.Linear(embedding_dim, embedding_dim)
        self.attn_k    = nn.Linear(embedding_dim * 3, embedding_dim)
        self.attn_v    = nn.Linear(embedding_dim * 3, embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, num_anchors * embedding_dim)
        self.norm      = nn.LayerNorm(embedding_dim)

    def forward(self, rel_embs, nbr_embs, time_deltas, mask):
        B, L, d = rel_embs.shape
        t_enc  = torch.tanh(self.time_proj(time_deltas))
        r_enc  = torch.relu(self.rel_proj(rel_embs))
        n_enc  = torch.relu(self.nbr_proj(nbr_embs))
        events = torch.cat([r_enc, n_enc, t_enc], dim=-1)

        denom  = mask.sum(dim=1, keepdim=True).clamp(min=1)
        query  = (events * mask.unsqueeze(-1)).sum(dim=1) / denom
        Q = self.attn_q(query[:, :d]).unsqueeze(1)
        K = self.attn_k(events)
        V = self.attn_v(events)

        attn = (Q * K).sum(dim=-1) / (d ** 0.5)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn.unsqueeze(-1) * V).sum(dim=1)
        out = self.norm(out)
        return self.out_proj(out).view(B, self.K, d)


class KARMA(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_anchors,
                 sigma=1.0, dropout=0.2, score_func='distmult', max_history=20):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d             = embedding_dim
        self.K             = num_anchors
        self.sigma         = sigma
        self.score_func    = score_func
        self.max_history   = max_history

        self.anchor_times = nn.Parameter(torch.randn(num_anchors))

        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        self.entity_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_table.weight)

        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        self.encoder    = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        self.cold_start = nn.Parameter(
            torch.empty(num_anchors, embedding_dim))
        nn.init.xavier_uniform_(self.cold_start)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def anchor_attention(self, tau):
        """tau: (B,) -> (B, K)"""
        diff  = tau.unsqueeze(1) - self.anchor_times.unsqueeze(0)
        return F.softmax(-diff.pow(2) / (self.sigma + 1e-8), dim=-1)

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau):
        """Pure inductive embedding via neighborhood encoder."""
        B, L = rel_ids.shape
        dev  = rel_ids.device

        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)

        mask = (torch.arange(L, device=dev).unsqueeze(0)
                < lengths.unsqueeze(1)).float()

        has_history = (lengths > 0).float().unsqueeze(-1)

        anchor_reps = self.encoder(rel_embs, nbr_embs, time_deltas, mask)
        alpha       = self.anchor_attention(tau)
        encoded     = (alpha.unsqueeze(-1) * anchor_reps).sum(dim=1)

        cold = (self.cold_start.unsqueeze(0) *
                alpha.unsqueeze(-1)).sum(dim=1)

        return has_history * encoded + (1.0 - has_history) * cold

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None):
        """
        KARMA-T: hybrid transductive + inductive embedding.
        Gate alpha is LEARNED — not just history length.
        Alpha=1 → fully inductive, Alpha=0 → fully transductive
        """
        ind_emb = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau)
        ind_emb = self.layer_norm(self.dropout(ind_emb))

        if entity_ids is None:
            return ind_emb

        trans_emb = self.layer_norm(
            self.dropout(self.entity_table(entity_ids)))

        hist_ratio = (lengths.float() / max(self.max_history, 1)).clamp(0, 1)
        gate_input = torch.cat([
            ind_emb, trans_emb,
            hist_ratio.unsqueeze(-1)
        ], dim=-1)
        alpha_gate = self.gate_net(gate_input)

        hybrid = alpha_gate * ind_emb + (1.0 - alpha_gate) * trans_emb
        return self.layer_norm(hybrid)

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus,
                s_eids=None, o_eids=None):

        s_emb = self.entity_embed(s_rids, s_nids, s_ts, s_lens, taus, s_eids)
        o_emb = self.entity_embed(o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        r_emb = self.dropout(self.relation_embeddings[r_ids])

        pos_scores = (s_emb * r_emb * o_emb).sum(dim=-1)
        return pos_scores
