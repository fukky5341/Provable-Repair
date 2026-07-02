import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from input_space.region import RegionStatus
from repair.bound import get_concrete_bounds
from repair.args import RepairTask
from .pareto.pareto import nondominated_rank

'''
Arachne
-------
arachne takes two components:
    - gradient loss
    - forward impact
--> pareto selection to get the score

outline
-------
goal: 
    1. get parameters that are less likely to cause deterioration of positive behavior, while improving the negative behavior
    2. select layer to repair

based on the score on positive and negative behavior individually, we select parameters and layers to repair
at least, we need to select parameters that aren't likely to deteriorate the positive behavior, which enables us to focus on parameters execpt them

given
-----
- repaired_net
- positive inputs
- negative region set
    - negative points are derived as corner, representative (likely to produce lower and upper bounds) points
- repaired layer index (optional)
- specification
- k_ratio (optional)
'''


'''helper functions'''
def _enable_target_layer_grads(target_layers):
    prev = []
    for layer in target_layers:
        w_prev = layer.weight.requires_grad
        b_prev = layer.bias.requires_grad if layer.bias is not None else None
        prev.append((layer, w_prev, b_prev))

        layer.weight.requires_grad_(True)
        if layer.bias is not None:
            layer.bias.requires_grad_(True)
    return prev

def _restore_target_layer_grads(prev):
    for layer, w_prev, b_prev in prev:
        layer.weight.requires_grad_(w_prev)
        if layer.bias is not None and b_prev is not None:
            layer.bias.requires_grad_(b_prev)


def collect_boundary_dataloader(model, dataloader, device, k=200, batch_size=32):
    model.eval()

    margins = []
    xs_all = []
    ys_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).long().view(-1)

            logits = model(x)

            true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            max_other = torch.topk(logits, 2, dim=1).values[:, 1]

            margin = true_logit - max_other

            margins.extend(margin.cpu().tolist())
            xs_all.append(x.cpu())
            ys_all.append(y.cpu())

    xs_all = torch.cat(xs_all)
    ys_all = torch.cat(ys_all)

    # sort by smallest margin
    idx = sorted(range(len(margins)), key=lambda i: margins[i])
    selected_idx = idx[:k]

    xs = xs_all[selected_idx]
    ys = ys_all[selected_idx]

    # create dataset + dataloader
    dataset = TensorDataset(xs, ys)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def collect_neg_dataloader(args, full_net, repaired_region_pairs, device, batch_size=32):
    xs = []
    ys = []

    for pair in repaired_region_pairs:
        for Nregion in pair.repaired_regions:

            if Nregion.status == RegionStatus.positive:
                continue

            label = Nregion.target_label
            out_C = Nregion.spec.C

            # ---- basic points ----
            if args.repair_task in ( RepairTask.CorruptionAndPerturbation, 
                                    RepairTask.LocalCounterexample,):
                xs.append(Nregion.center_point.cpu())
                ys.append(torch.tensor([label]))

            Nlb = Nregion.lb.unsqueeze(0).to(device)
            Nub = Nregion.ub.unsqueeze(0).to(device)

            xs.append(Nlb.cpu())
            ys.append(torch.tensor([label]))

            xs.append(Nub.cpu())
            ys.append(torch.tensor([label]))

            # ---- bounds + coeff ----
            lbs, ubs, bounder = get_concrete_bounds(full_net, Nlb.squeeze(0), Nub.squeeze(0), save_coeffs=True)
            output_coeff = bounder.saved_coeffs[(len(full_net)-1, 0)]

            rlb = Nlb.view(-1)
            rub = Nub.view(-1)

            # ---- target dim and violating dims ----
            out_ub = ubs[-1].view(-1)
            out_lb = lbs[-1].view(-1)
            # violating dim
            pos_C = torch.clamp(out_C, min=0)
            target_dim_lb = (pos_C[0] * out_lb).sum()
            dims = (target_dim_lb - out_ub) < 0
            dims = dims.nonzero(as_tuple=True)[0]

            for vdim in dims:
                coeff_lb_mv = output_coeff.n_C_lb[vdim].view(-1)
                coeff_ub_mv = output_coeff.n_C_ub[vdim].view(-1)

                corner_lb_mv = torch.where(coeff_lb_mv >= 0, rlb, rub).view_as(Nlb)
                corner_ub_mv = torch.where(coeff_ub_mv >= 0, rlb, rub).view_as(Nub)

                xs.append(corner_lb_mv.cpu())
                ys.append(torch.tensor([label]))

                xs.append(corner_ub_mv.cpu())
                ys.append(torch.tensor([label]))

    if len(xs) == 0:
        raise ValueError("No negative points were collected.")

    xs_tensor = torch.cat(xs, dim=0)
    ys_tensor = torch.cat(ys, dim=0).long()

    dataset = TensorDataset(xs_tensor, ys_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

'''
gradient loss
-------------
dL/dw
'''

def compute_positive_grad_full(
    repaired_net,
    target_layers,
    pos_loader,
    device,
    loss_type="logit",
):
    repaired_net.eval()
    prev_grad_flags = _enable_target_layer_grads(target_layers)

    # initialize storage
    grad_w = {}
    grad_b = {}
    for layer in target_layers:
        grad_w[layer] = torch.zeros_like(layer.weight)
        if layer.bias is not None:
            grad_b[layer] = torch.zeros_like(layer.bias)

    # cache original outputs (no grad)
    orig_outputs = []
    with torch.no_grad():
        for x, _ in pos_loader:
            x = x.to(device)
            orig_outputs.append(repaired_net(x))

    # iterate again for gradient computation
    for (x, y), y_orig in zip(pos_loader, orig_outputs):
        x = x.to(device)
        y = y.to(device).long().view(-1)
        y_orig = y_orig.to(device)

        repaired_net.zero_grad()
        out = repaired_net(x)

        if loss_type == "logit":
            loss = ((out - y_orig) ** 2).mean()
        elif loss_type == "ce":
            loss = F.cross_entropy(out, y)
        else:
            raise ValueError("Unknown loss_type")

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Target layers may be frozen.")
        loss.backward()

        # ---- collect gradients for each target layer ----
        for layer in target_layers:
            if layer.weight.grad is not None:
                grad_w[layer] += layer.weight.grad.detach().abs()
            if layer.bias is not None and layer.bias.grad is not None:
                grad_b[layer] += layer.bias.grad.detach().abs()
    
    num_batches = len(pos_loader)
    for layer in grad_w:
        grad_w[layer] /= num_batches
        if layer in grad_b:
            grad_b[layer] /= num_batches

    _restore_target_layer_grads(prev_grad_flags)
    return grad_w, grad_b


'''
forward impact
--------------
(weight)
example of the connection between oi -- wij --> o'j -- relu --> ...
1. the importance of wij in incoming connection relating to the o'j
    --> oi * wij normalized by the sum of incoming connection (sum_k ok * wkj)
2. the gradient of o'j with respect to output loss
    --> dO/do'j
3. total importance
    --> normalized(oi * wij) * dO/do'j

(bias)
bias is simpler since it doesn't have the importance of incoming connection
--> dO/do'j
'''

def arachne_forward_importance_layer_(layer, activation, grad_out):
    """
    layer: nn.Linear or Conv2d
    activation: input to layer (batch, in_features, ...)
    grad_out: gradient wrt pre-activation (batch, out_features, ...)
    """

    W = layer.weight

    # flatten spatial if needed
    if activation.dim() > 2:
        activation = activation.view(activation.shape[0], activation.shape[1], -1).mean(dim=2)
        grad_out = grad_out.view(grad_out.shape[0], grad_out.shape[1], -1).mean(dim=2)

    # average over batch
    a = activation.mean(dim=0)        # (in,)
    g = grad_out.mean(dim=0)          # (out,)

    # contribution
    contrib = a.unsqueeze(0) * W      # (out, in)

    denom = contrib.abs().sum(dim=1, keepdim=True) + 1e-8
    norm_contrib = contrib / denom

    importance = norm_contrib * g.unsqueeze(1)
    importance_b = g

    return importance.abs(), importance_b.abs()


def arachne_forward_importance_layer(layer, activation, grad_out):
    W = layer.weight  # (out, in, kh, kw)

    if isinstance(layer, torch.nn.Linear):
        # ---- same as before ----
        a = activation.mean(dim=0)   # (in,)
        g = grad_out.mean(dim=0)     # (out,)

        contrib = a.unsqueeze(0) * W
        denom = contrib.abs().sum(dim=1, keepdim=True) + 1e-8
        norm_contrib = contrib / denom

        importance = norm_contrib * g.unsqueeze(1)
        importance_b = g

        return importance.abs(), importance_b.abs()

    elif isinstance(layer, torch.nn.Conv2d):
        B, Cin, Hin, Win = activation.shape
        Cout, _, Kh, Kw = W.shape

        # ---- unfold input into patches ----
        patches = F.unfold(
            activation,
            kernel_size=(Kh, Kw),
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation
        )  # (B, Cin*Kh*Kw, L)

        # reshape weight
        W_flat = W.view(Cout, -1)  # (Cout, Cin*Kh*Kw)

        # flatten grad_out spatially
        grad_flat = grad_out.view(B, Cout, -1)  # (B, Cout, L)

        # ---- compute contribution ----
        # patches: (B, K, L)
        # W_flat: (Cout, K)
        # grad_flat: (B, Cout, L)

        # contribution per weight
        weighted_patches = patches.unsqueeze(1) * W_flat.unsqueeze(0).unsqueeze(-1)
        # (B, Cout, K, L)

        contrib = (weighted_patches * grad_flat.unsqueeze(2)).sum(dim=(0, 3))
        # (Cout, K)

        # normalize per output channel
        denom = contrib.abs().sum(dim=1, keepdim=True) + 1e-8
        norm_contrib = contrib / denom

        importance = norm_contrib.view_as(W)

        # bias importance
        importance_b = grad_out.sum(dim=(0, 2, 3))  # (Cout,)

        return importance.abs(), importance_b.abs()

    else:
        raise NotImplementedError("Unsupported layer type")


def compute_forward_impact_positive(
    repaired_net,
    target_layers,
    pos_loader,
    device,
    loss_type="logit"
):
    fwd_w = {layer: torch.zeros_like(layer.weight) for layer in target_layers}
    fwd_b = {layer: torch.zeros_like(layer.bias) for layer in target_layers if layer.bias is not None}

    repaired_net.eval()
    prev_grad_flags = _enable_target_layer_grads(target_layers)

    # register hooks (as above)
    activations = {}
    grad_outputs = {}
    hooks = []

    for layer in target_layers:
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, layer=layer: activations.update({layer: inp[0]})
        ))
        hooks.append(layer.register_full_backward_hook(
            lambda m, gin, gout, layer=layer: grad_outputs.update({layer: gout[0]})
        ))

    # precompute outside loop
    orig_outputs = []
    with torch.no_grad():
        for x, _ in pos_loader:
            orig_outputs.append(repaired_net(x))

    total_samples = 0
    for (x, y), y_orig in zip(pos_loader, orig_outputs):
        x = x.to(device)
        y = y.to(device).long().view(-1)
        y_orig = y_orig.to(device)

        total_samples += x.size(0)

        repaired_net.zero_grad()
        out = repaired_net(x)

        if loss_type == "logit":
            loss = ((out - y_orig) ** 2).mean()
        elif loss_type == "ce":
            loss = F.cross_entropy(out, y)
        else:
            raise ValueError("Unknown loss_type")

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Target layers may be frozen.")
        loss.backward()

        for layer in target_layers:
            if layer in activations and layer in grad_outputs:
                importance_w, importance_b = arachne_forward_importance_layer(
                    layer,
                    activations[layer].detach(),
                    grad_outputs[layer].detach()
                )
                fwd_w[layer] += importance_w
                if layer.bias is not None:
                    fwd_b[layer] += importance_b

        activations.clear()
        grad_outputs.clear()

    for h in hooks:
        h.remove()

    for layer in fwd_w:
        fwd_w[layer] /= total_samples
        if layer in fwd_b:
            fwd_b[layer] /= total_samples

    _restore_target_layer_grads(prev_grad_flags)
    return fwd_w, fwd_b

def compute_forward_impact_negative(
    repaired_net,
    target_layers,
    neg_loader,
    device,
    loss_type="margin"
):
    fwd_w = {layer: torch.zeros_like(layer.weight) for layer in target_layers}
    fwd_b = {layer: torch.zeros_like(layer.bias) for layer in target_layers if layer.bias is not None}

    repaired_net.eval()
    prev_grad_flags = _enable_target_layer_grads(target_layers)

    activations = {}
    grad_outputs = {}
    hooks = []

    for layer in target_layers:
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, layer=layer: activations.update({layer: inp[0]})
        ))
        hooks.append(layer.register_full_backward_hook(
            lambda m, gin, gout, layer=layer: grad_outputs.update({layer: gout[0]})
        ))

    total_samples = 0

    for x, y in neg_loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)

        repaired_net.zero_grad()
        out = repaired_net(x)

        if loss_type == "margin":
            true_logit = out.gather(1, y.unsqueeze(1)).squeeze(1)
            tmp = out.clone()
            tmp[torch.arange(out.size(0), device=out.device), y] = -1e9
            max_other = tmp.max(dim=1).values
            loss = (max_other - true_logit).mean()
        else:
            raise ValueError("Unknown loss_type")

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Target layers may be frozen.")
        loss.backward()

        B = x.size(0)
        total_samples += B

        for layer in target_layers:
            if layer in activations and layer in grad_outputs:
                importance_w, importance_b = arachne_forward_importance_layer(
                    layer,
                    activations[layer].detach(),
                    grad_outputs[layer].detach()
                )
                fwd_w[layer] += importance_w * B
                if layer.bias is not None:
                    fwd_b[layer] += importance_b * B

        activations.clear()
        grad_outputs.clear()

    for h in hooks:
        h.remove()

    for layer in fwd_w:
        fwd_w[layer] /= total_samples
        if layer in fwd_b:
            fwd_b[layer] /= total_samples

    _restore_target_layer_grads(prev_grad_flags)
    return fwd_w, fwd_b


'''
paleto selection
-----------------
we have:
    - gradient importance for positive behavior (grad_pos)
        --> minimize to avoid touching risky parameters
    - gradient importance for negative behavior (grad_neg)
        --> maximize to fix negative behavior
    - forward impact for positive behavior (fwd_pos)
        --> minimize to avoid deteriorating positive behavior
    - forward impact for negative behavior (fwd_neg)
        --> maximize to fix negative behavior

1. normalize each importance
2. add sign for each importance based on whether we want to maximize or minimize
3. pareto selection based on the 4 scores
'''

def build_score_matrix4d(grad_pos, grad_neg, fwd_pos, fwd_neg):
    # flatten
    g_pos = grad_pos.view(-1)
    g_neg = grad_neg.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)

    # normalize (important!)
    def norm(x):
        return x / (x.mean() + 1e-8)

    g_pos = norm(g_pos)
    g_neg = norm(g_neg)
    f_pos = norm(f_pos)
    f_neg = norm(f_neg)

    # maximize all
    scores = torch.stack([
        -g_pos,    # minimize positive gradient importance
        g_neg,     # maximize negative gradient importance
        -f_pos,    # minimize positive forward impact
        f_neg      # maximize negative forward impact
    ], dim=1)

    return scores

def build_score_matrix2d(grad_pos, grad_neg, fwd_pos, fwd_neg):
    # flatten
    g_pos = grad_pos.view(-1)
    g_neg = grad_neg.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)

    # normalize (important!)
    def norm(x):
        return x / (x.mean() + 1e-8)

    g_pos = norm(g_pos)
    g_neg = norm(g_neg)
    f_pos = norm(f_pos)
    f_neg = norm(f_neg)

    # combine into 2D score (for visualization and simpler selection)
    score_pos = - g_pos - f_pos
    score_neg = g_neg + f_neg

    scores = torch.stack([score_pos, score_neg], dim=1)

    return scores

def pareto_front(scores, eps=1e-2):
    """
    scores: (N, K) tensor
    returns: boolean mask (N,)
    """
    N = scores.shape[0]
    keep = torch.ones(N, dtype=torch.bool, device=scores.device)

    for i in range(N):
        if not keep[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if torch.all(scores[j] >= scores[i] - eps) and torch.any(scores[j] > scores[i] + eps):
                keep[i] = False
                break
    return keep

def pareto_selection_with_fallback(grad_neg, grad_pos, fwd_pos, fwd_neg, k_ratio=0.1):
    if k_ratio is None:
        pre_ratio = 1.0
    else:
        pre_ratio = min(1.0, k_ratio + 0.3)

    flat_neg = grad_neg.abs().view(-1)
    # pre-filtering
    k = max(1, int(pre_ratio * flat_neg.numel()))
    topk = torch.topk(flat_neg, k).indices

    # score matrix
    scores = build_score_matrix2d(grad_pos, grad_neg, fwd_pos, fwd_neg)
    scores_filtered = scores[topk]

    # pareto selection
    mask = pareto_front(scores_filtered)

    # map back
    mask_flat = torch.zeros_like(flat_neg, dtype=torch.bool)
    mask_flat[topk[mask]] = True

    return mask_flat.view_as(grad_neg)

def multi_pareto_selection(
    grad_neg, grad_pos, fwd_pos, fwd_neg,
    k_ratio=0.01,
    max_rounds=3,
    prefilter=False
):
    flat_size = grad_neg.numel()
    k_total = max(1, int(k_ratio * flat_size))

    # ---- scores (full space) ----
    scores = build_score_matrix2d(
        grad_pos, grad_neg, fwd_pos, fwd_neg
    ).view(flat_size, -1)

    # =====================
    # Pre-filter (reduce space)
    # =====================
    if prefilter:
        prefilter_ratio = min(1.0, k_ratio)
        k_pref = max(1, int(prefilter_ratio * flat_size))
        tmp_scores = (grad_neg + fwd_neg) - (grad_pos + fwd_pos)

        topk_pref = torch.topk(tmp_scores, k_pref).indices

        scores_pref = scores[topk_pref]
    else:
        k_pref = flat_size
        topk_pref = torch.arange(flat_size, device=grad_neg.device)
        scores_pref = scores

    # =====================
    # Work in reduced space
    # =====================
    selected_small = torch.zeros(k_pref, dtype=torch.bool, device=grad_neg.device)
    remaining_small = torch.ones(k_pref, dtype=torch.bool, device=grad_neg.device)

    for _ in range(max_rounds):

        if selected_small.sum() >= k_total:
            break

        idx = torch.where(remaining_small)[0]
        if len(idx) == 0:
            break

        scores_subset = scores_pref[idx]

        pareto_mask = pareto_front(scores_subset)

        selected_idx = idx[pareto_mask]

        selected_small[selected_idx] = True
        remaining_small[selected_idx] = False

    # =====================
    # Map back to full space
    # =====================
    selected_mask = torch.zeros(flat_size, dtype=torch.bool, device=grad_neg.device)
    selected_mask[topk_pref[selected_small]] = True

    return selected_mask.view_as(grad_neg)

def fast_pareto_selection(
    grad_neg, grad_pos, fwd_pos, fwd_neg,
    k_ratio=0.01,
    prefilter_ratio=None
):
    device = grad_neg.device
    flat_size = grad_neg.numel()
    k_total = max(1, int(k_ratio * flat_size))

    def norm(x):
        return x / (x.mean() + 1e-8)

    # ---- flatten ----
    g_neg = grad_neg.view(-1)
    g_pos = grad_pos.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)
    # ---- normalize ----
    g_neg = norm(g_neg)
    g_pos = norm(g_pos)
    f_pos = norm(f_pos)
    f_neg = norm(f_neg)

    # =====================
    # Optional prefilter (strongly recommended)
    # =====================
    if prefilter_ratio is None:
        prefilter_ratio = min(1.0, k_ratio + 0.1)

    k_pref = max(1, int(prefilter_ratio * flat_size))

    # fast scalar score for filtering
    score_fast = (g_neg + f_neg) - (g_pos + f_pos)  # higher is better 
    topk_idx = torch.topk(score_fast, k_pref).indices

    # restrict
    g_neg = g_neg[topk_idx]
    g_pos = g_pos[topk_idx]
    f_pos = f_pos[topk_idx]
    f_neg = f_neg[topk_idx]

    # =====================
    # Build 2D score (Pareto)
    # =====================
    # NOTE: in nondominated_rank, scores are designed so that smaller is better
    # convert to minimization form
    score_pos = g_pos + f_pos          # minimize
    score_neg = -(g_neg + f_neg)       # maximize → minimize

    scores = torch.stack([score_pos, score_neg], dim=1)

    # =====================
    # Fast Pareto ranking
    # =====================
    ranks = nondominated_rank(scores.detach().cpu().numpy())
    ranks = torch.from_numpy(ranks).to(device)

    # =====================
    # Select top-k by rank
    # =====================
    # smaller rank = better
    sorted_idx = torch.argsort(ranks)

    selected_small = sorted_idx[:k_total]

    # =====================
    # map back to full space
    # =====================
    selected_mask = torch.zeros(flat_size, dtype=torch.bool, device=device)
    selected_mask[topk_idx[selected_small]] = True

    return selected_mask.view_as(grad_neg)

def two_stage_pareto(
    grad_neg, grad_pos, fwd_pos, fwd_neg,
    k_ratio=0.01,
    prefilter_ratio=None,
    pareto_round=3
):
    """
    1. fast pareto
    2. multi pareto
    """
    device = grad_neg.device
    flat_size = grad_neg.numel()

    # flatten
    g_neg = grad_neg.view(-1)
    g_pos = grad_pos.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)
    
    # stage 1: fast pareto
    mask_fast = fast_pareto_selection(
        grad_neg, grad_pos, fwd_pos, fwd_neg,
        k_ratio=k_ratio,
        prefilter_ratio=prefilter_ratio
    ).view(-1)
    idx_fast = torch.where(mask_fast)[0]

    # stage 2: multi pareto
    # ---- select only the fast pareto ones ----
    grad_neg_sel = g_neg[idx_fast]
    grad_pos_sel = g_pos[idx_fast]
    fwd_pos_sel = f_pos[idx_fast]
    fwd_neg_sel = f_neg[idx_fast]
    # ---- multi pareto selection ----
    mask_multi_sel = multi_pareto_selection(
        grad_neg_sel, grad_pos_sel, fwd_pos_sel, fwd_neg_sel,
        k_ratio=1.0,  # use all since we already reduced by fast pareto
        max_rounds=pareto_round,
        prefilter=False
    ).view(-1)

    # =====================
    # map back
    # =====================
    selected_mask = torch.zeros(flat_size, dtype=torch.bool, device=device)
    selected_mask[idx_fast[mask_multi_sel]] = True

    return selected_mask.view_as(grad_neg)


def arachne_selection(args, full_net, repaired_layer, grad_neg_w, grad_neg_b, tr_loader, repaired_region_pairs, last_layer_repair=False):
    '''Note: currently only supports one layer (repaired_layer)'''
    # prepare positive loader (boundary samples) with 200 samples (hyperparameter)
    pos_loader = collect_boundary_dataloader(full_net, tr_loader, device=args.device, k=200)
    # compute grad_pos, fwd_pos
    grad_pos_w, grad_pos_b = compute_positive_grad_full(full_net, [repaired_layer], pos_loader, device=grad_neg_w.device)
    fwd_pos_w, fwd_pos_b = compute_forward_impact_positive(full_net, [repaired_layer], pos_loader, device=grad_neg_w.device)
    # prepare negative set
    neg_loader = collect_neg_dataloader(args, full_net, repaired_region_pairs, device=args.device)
    fwd_neg_w, fwd_neg_b = compute_forward_impact_negative(full_net, [repaired_layer], neg_loader, device=grad_neg_w.device)

    if last_layer_repair:
        pareto_round = 3  # todo
    else:
        pareto_round = 3
    # pareto selection
    # mask_w = multi_pareto_selection(grad_neg_w, grad_pos_w[repaired_layer], fwd_pos_w[repaired_layer], fwd_neg_w[repaired_layer], k_ratio=args.fl_k_ratio, max_rounds=pareto_round)
    mask_w = two_stage_pareto(grad_neg_w, grad_pos_w[repaired_layer], fwd_pos_w[repaired_layer], fwd_neg_w[repaired_layer], k_ratio=args.fl_k_ratio, pareto_round=args.pareto_round)
    if repaired_layer.bias is not None:
        # mask_b = multi_pareto_selection(grad_neg_b, grad_pos_b[repaired_layer], fwd_pos_b[repaired_layer], fwd_neg_b[repaired_layer], k_ratio=args.fl_k_ratio, max_rounds=pareto_round)
        mask_b = two_stage_pareto(grad_neg_b, grad_pos_b[repaired_layer], fwd_pos_b[repaired_layer], fwd_neg_b[repaired_layer], k_ratio=args.fl_k_ratio, pareto_round=args.pareto_round)
    else:
        mask_b = None

    # # two-stage selection
    # mask_w = two_stage_selection(grad_neg_w, grad_pos_w[repaired_layer], fwd_pos_w[repaired_layer], fwd_neg_w[repaired_layer], k_ratio=args.fl_k_ratio)
    # if repaired_layer.bias is not None:
    #     mask_b = two_stage_selection(grad_neg_b, grad_pos_b[repaired_layer], fwd_pos_b[repaired_layer], fwd_neg_b[repaired_layer], k_ratio=args.fl_k_ratio)
    # else:
    #     mask_b = None

    return mask_w, mask_b

