import os, argparse, json, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import KARMA
from data_loader import TKGDataset, get_dataloaders, build_filter_dict_normalized
from evaluate import evaluate, print_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',       type=str,   required=True)
    p.add_argument('--data_dir',      type=str,   default='/storage/bhull113/rhgnn/data')
    p.add_argument('--save_dir',      type=str,   default='/storage/bhull113/karma/checkpoints')
    p.add_argument('--embedding_dim', type=int,   default=200)
    p.add_argument('--num_anchors',   type=int,   default=16)
    p.add_argument('--num_clusters',  type=int,   default=4)
    p.add_argument('--sigma',         type=float, default=0.5)
    p.add_argument('--score_func',    type=str,   default='distmult')
    p.add_argument('--dropout',       type=float, default=0.2)
    p.add_argument('--lambda2',       type=float, default=0.001)
    p.add_argument('--lambda3',       type=float, default=0.01)
    p.add_argument('--num_neg',       type=int,   default=64)
    p.add_argument('--epochs',        type=int,   default=500)
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-5)
    p.add_argument('--patience',      type=int,   default=20)
    p.add_argument('--eval_every',    type=int,   default=5)
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--max_history',   type=int,   default=20)
    p.add_argument('--gpu',           type=int,   default=0)
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cluster_diversity_loss(cluster_w):
    """Maximize entropy of cluster assignments — prevent collapse to one cluster."""
    mean_usage = cluster_w.mean(dim=0)                          # (C,)
    entropy    = -(mean_usage * (mean_usage + 1e-8).log()).sum()
    return -entropy


def anchor_spread_loss(model):
    """Keep anchors spread across time within and between clusters."""
    loss = torch.tensor(0.0,
        device=model.adaptive_anchor.anchor_clusters.device)

    for c in range(model.C):
        anchors = model.adaptive_anchor.anchor_clusters[c]
        if len(anchors) > 1:
            spread = torch.pdist(anchors.unsqueeze(1), p=2).mean()
            loss   = loss + torch.clamp(1.0 - spread, min=0)

    cluster_means = model.adaptive_anchor.anchor_clusters.mean(dim=1)
    if len(cluster_means) > 1:
        diversity = torch.pdist(cluster_means.unsqueeze(1), p=2).mean()
        loss      = loss + torch.clamp(1.0 - diversity, min=0)

    return loss


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  KARMA-A  |  Dataset: {args.dataset}  |  Device: {device}")
    print(f"  Anchors: {args.num_anchors}  Clusters: {args.num_clusters}")
    print(f"{'='*60}")

    data_path = os.path.join(args.data_dir, args.dataset)
    tkgdata   = TKGDataset(data_path)

    train_loader, valid_loader, test_loader = get_dataloaders(
        tkgdata, batch_size=args.batch_size, num_workers=args.num_workers)

    all_quads   = np.concatenate([tkgdata.train, tkgdata.valid, tkgdata.test], axis=0)
    filter_dict = build_filter_dict_normalized(
        all_quads, tkgdata.t_min, tkgdata.t_range)

    model = KARMA(
        num_entities  = tkgdata.num_entities,
        num_relations = tkgdata.num_relations,
        embedding_dim = args.embedding_dim,
        num_anchors   = args.num_anchors,
        num_clusters  = args.num_clusters,
        sigma         = args.sigma,
        dropout       = args.dropout,
        score_func    = args.score_func,
        max_history   = args.max_history,
    ).to(device)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path    = os.path.join(args.save_dir, f'karmaa_{args.dataset}_best.pt')
    results_path = os.path.join(args.save_dir, f'karmaa_{args.dataset}_results.json')

    best_mrr, patience_counter = 0.0, 0
    N = tkgdata.num_entities

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        for batch in train_loader:
            (s_ids, r_ids, o_ids, taus,
             s_rels, s_nbrs, s_times, s_lens,
             o_rels, o_nbrs, o_times, o_lens) = [x.to(device) for x in batch]

            optimizer.zero_grad()
            B = r_ids.size(0)

            # KARMA-A hybrid embeddings with cluster weights
            s_result = model.entity_embed(
                s_rels, s_nbrs, s_times, s_lens, taus, s_ids)
            o_result = model.entity_embed(
                o_rels, o_nbrs, o_times, o_lens, taus, o_ids)

            s_embs, s_cw = s_result
            o_embs, o_cw = o_result
            r_embs = model.relation_embeddings[r_ids]

            # Positive scores
            pos_scores = (s_embs * r_embs * o_embs).sum(dim=-1)

            # Negative via entity_table
            neg_ids    = torch.randint(0, N, (B, args.num_neg), device=device)
            neg_embs   = model.layer_norm(model.entity_table(neg_ids))
            sr_exp     = (s_embs * r_embs).unsqueeze(1)
            neg_scores = (sr_exp * neg_embs).sum(dim=-1)

            pos_nb    = model.layer_norm(model.entity_table(o_ids))
            pos_nb_sc = (s_embs * r_embs * pos_nb).sum(dim=-1)

            # BCE loss
            loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                pos_nb_sc, torch.ones(B, device=device))
            loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros(B, args.num_neg, device=device))
            loss = loss_pos + loss_neg

            # KARMA-A: cluster diversity loss
            div_loss = cluster_diversity_loss(s_cw) + \
                       cluster_diversity_loss(o_cw)
            loss = loss + args.lambda3 * div_loss

            # Anchor spread loss
            spread_loss = anchor_spread_loss(model)
            loss = loss + args.lambda2 * spread_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")

        if epoch % args.eval_every == 0:
            vr = evaluate(model, valid_loader, filter_dict,
                          tkgdata.num_entities, device, 'valid')
            scheduler.step(vr['mrr'])
            tag = ''
            if vr['mrr'] > best_mrr:
                best_mrr, patience_counter = vr['mrr'], 0
                torch.save({'epoch': epoch,
                            'model_state': model.state_dict(),
                            'args': vars(args),
                            'valid_results': vr}, ckpt_path)
                tag = '  *** saved ***'
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            print(f"  Val MRR:{vr['mrr']:.4f} H@1:{vr['hits1']:.4f} "
                  f"H@3:{vr['hits3']:.4f} H@10:{vr['hits10']:.4f}{tag}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_results = evaluate(
        model, test_loader, filter_dict,
        tkgdata.num_entities, device, 'test',
        unseen_entities=tkgdata.unseen_entities,
    )
    print_results(test_results, split=f'Test [{args.dataset}] KARMA-A')

    with open(results_path, 'w') as f:
        json.dump({
            'dataset':      args.dataset,
            'model':        'KARMA-A',
            'best_epoch':   ckpt['epoch'],
            'valid':        ckpt['valid_results'],
            'test':         test_results,
            'args':         vars(args),
            'unseen_count': len(tkgdata.unseen_entities),
            'seen_count':   len(tkgdata.seen_entities),
        }, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == '__main__':
    main()
