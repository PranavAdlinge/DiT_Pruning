import math
import torch
import torch.nn as nn


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = {}
    for rank_list in rankings:
        for rank, item in enumerate(rank_list):
            if item not in fused_scores:
                fused_scores[item] = 0
            fused_scores[item] += 1 / (k + rank + 1)
    return fused_scores


class Flux2StructuredLinearPruner(object):
    def __init__(self, layer, args):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = self.layer.weight.shape[0]
        self.columns = self.layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
        self.sum_weight = 0
        self.no_compensate = args.no_compensate

    def add_batch(self, inp, out, weight_new):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        weight_old = self.sum_weight
        weight_total = weight_old + weight_new
        self.H *= weight_old / weight_total
        self.sum_weight = weight_total

        norm_factor = math.sqrt(2 / self.sum_weight)
        inp = norm_factor * inp.float()
        self.H += inp.matmul(inp.t())

    def _prepare_weight_and_hessian(self, percdamp):
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        return W, H

    def _apply_selected_columns(self, W, H, Hinv, selected_idx):
        selected_idx = torch.sort(selected_idx).values
        selected_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
        selected_mask[selected_idx] = True

        all_idx = torch.arange(self.columns, device=self.dev)
        active_mask = ~(self.column_mask | selected_mask)
        pruned_mask = self.column_mask
        column_sort_idx = torch.cat([all_idx[selected_mask], all_idx[active_mask], all_idx[pruned_mask]])

        W = W[:, column_sort_idx]
        Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
        Hinv = torch.linalg.cholesky(Hinv, upper=True)[: selected_idx.numel()]

        W_pruned = W[:, : selected_idx.numel()].clone()
        Hinv_pruned = Hinv[:, : selected_idx.numel()]
        Err = torch.zeros_like(W_pruned)

        for i in range(selected_idx.numel()):
            Err[:, i : i + 1] = W_pruned[:, i : i + 1] / Hinv_pruned[i, i]
            if not self.no_compensate:
                W_pruned[:, i:] -= Err[:, i : i + 1].matmul(Hinv_pruned[i : i + 1, i:])

        W[:, : selected_idx.numel()] = 0
        if not self.no_compensate:
            active_columns = active_mask.count_nonzero().item()
            end = selected_idx.numel() + active_columns
            W[:, selected_idx.numel() : end] -= Err.matmul(Hinv[:, selected_idx.numel() : end])

        column_sort_idx_inv = torch.argsort(column_sort_idx)
        W = W[:, column_sort_idx_inv]

        H[selected_idx, :] = 0
        H[:, selected_idx] = 0
        H[selected_idx, selected_idx] = 1
        self.H[selected_idx, :] = 0
        self.H[:, selected_idx] = 0
        self.H[selected_idx, selected_idx] = 1
        self.column_mask[selected_idx] = True

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        return selected_idx

    def struct_prune(self, sparsity, percdamp=0.0, group_size=1, candidate_idx=None, blocksize=128):
        if candidate_idx is None:
            candidate_idx = torch.arange(self.columns, device=self.dev)
        else:
            candidate_idx = torch.as_tensor(candidate_idx, device=self.dev, dtype=torch.long)

        if candidate_idx.numel() == 0 or sparsity <= 0:
            return torch.empty(0, dtype=torch.long, device=self.dev)

        W, H = self._prepare_weight_and_hessian(percdamp)
        candidate_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
        candidate_mask[candidate_idx] = True

        if group_size > 1:
            assert candidate_idx.numel() % group_size == 0
            target_groups = round(candidate_idx.numel() // group_size * sparsity)
            target_columns = target_groups * group_size
        else:
            target_columns = round(candidate_idx.numel() * sparsity)

        if target_columns == 0:
            return torch.empty(0, dtype=torch.long, device=self.dev)

        pruned_columns = 0
        pruned_indices = []

        while pruned_columns < target_columns:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

            if group_size > 1:
                groups = candidate_idx.view(-1, group_size)
                group_scores = []
                for group in groups:
                    if self.column_mask[group].any():
                        group_scores.append(torch.tensor(torch.inf, device=self.dev))
                        continue

                    group_hinv = Hinv[group][:, group]
                    chol_diag = torch.diagonal(torch.linalg.cholesky(group_hinv), dim1=-2, dim2=-1)
                    score = torch.sum(W[:, group] ** 2 / chol_diag.pow(2).unsqueeze(0))
                    group_scores.append(score)

                best_group = groups[torch.stack(group_scores).argmin()]
                selected_idx = best_group
            else:
                Hinv_diag = Hinv.diag()
                error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
                error[~candidate_mask] = torch.inf
                error[self.column_mask] = torch.inf

                remaining_candidates = (~self.column_mask & candidate_mask).count_nonzero().item()
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024, remaining_candidates)
                selected_idx = error.argsort()[:cnt]

            selected_idx = self._apply_selected_columns(W, H, Hinv, selected_idx)
            pruned_indices.append(selected_idx)
            W = self.layer.weight.data.clone().float()
            pruned_columns += selected_idx.numel()

        pruned_indices = torch.cat(pruned_indices) if pruned_indices else torch.empty(0, dtype=torch.long, device=self.dev)
        print(f"pruned columns {pruned_indices.numel()}/{candidate_idx.numel()} on {self.layer.__class__.__name__}")
        return torch.sort(pruned_indices).values

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.column_mask = None
        torch.cuda.empty_cache()


class Flux2StructuredJointAttentionPruner(object):
    def __init__(self, layer_to_out, layer_to_add_out, args):
        self.layer_to_out = layer_to_out
        self.layer_to_add_out = layer_to_add_out
        self.dev = self.layer_to_out.weight.device

        self.to_out_columns = self.layer_to_out.weight.shape[1]
        self.to_add_out_columns = self.layer_to_add_out.weight.shape[1]
        self.H_to_out = torch.zeros((self.to_out_columns, self.to_out_columns), device=self.dev)
        self.H_to_add_out = torch.zeros((self.to_add_out_columns, self.to_add_out_columns), device=self.dev)
        self.column_mask = torch.zeros(self.to_out_columns, dtype=torch.bool, device=self.dev)
        self.sum_weight_to_out = 0
        self.sum_weight_to_add_out = 0
        self.no_compensate = args.no_compensate
        self.percdamp = args.percdamp

    def add_batch(self, inp, out, layer_name, weight_new):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if layer_name == "attn.to_out.0":
            weight_old = self.sum_weight_to_out
            weight_total = weight_old + weight_new
            self.H_to_out *= weight_old / weight_total
            self.sum_weight_to_out = weight_total
            norm_factor = math.sqrt(2 / self.sum_weight_to_out)
            inp = norm_factor * inp.float()
            self.H_to_out += inp.matmul(inp.t())
        elif layer_name == "attn.to_add_out":
            weight_old = self.sum_weight_to_add_out
            weight_total = weight_old + weight_new
            self.H_to_add_out *= weight_old / weight_total
            self.sum_weight_to_add_out = weight_total
            norm_factor = math.sqrt(2 / self.sum_weight_to_add_out)
            inp = norm_factor * inp.float()
            self.H_to_add_out += inp.matmul(inp.t())

    def struct_prune(self, sparsity, headsize=1, percdamp=0.0):
        assert self.to_out_columns == self.to_add_out_columns
        assert self.to_out_columns % headsize == 0

        W1 = self.layer_to_out.weight.data.clone().float()
        W2 = self.layer_to_add_out.weight.data.clone().float()
        H1 = self.H_to_out
        H2 = self.H_to_add_out

        dead_1 = torch.diag(H1) == 0
        H1[dead_1, dead_1] = 1
        W1[:, dead_1] = 0

        dead_2 = torch.diag(H2) == 0
        H2[dead_2, dead_2] = 1
        W2[:, dead_2] = 0

        if percdamp > 0:
            damp_1 = percdamp * torch.mean(torch.diag(H1))
            diag_1 = torch.arange(H1.size(0), device=self.dev)
            H1[diag_1, diag_1] += damp_1

            damp_2 = percdamp * torch.mean(torch.diag(H2))
            diag_2 = torch.arange(H2.size(0), device=self.dev)
            H2[diag_2, diag_2] += damp_2

        target_columns = round(self.to_out_columns // headsize * sparsity) * headsize
        pruned_columns = 0

        while pruned_columns < target_columns:
            Hinv_1 = torch.cholesky_inverse(torch.linalg.cholesky(H1))
            Hinv_2 = torch.cholesky_inverse(torch.linalg.cholesky(H2))

            Hinv_diag_1 = torch.stack(
                [Hinv_1[i : i + headsize, i : i + headsize] for i in range(0, self.to_out_columns, headsize)]
            )
            Hinv_diag_1 = torch.diagonal(torch.linalg.cholesky(Hinv_diag_1), dim1=-2, dim2=-1).reshape(-1)
            Hinv_diag_1 = Hinv_diag_1 ** 2

            Hinv_diag_2 = torch.stack(
                [Hinv_2[i : i + headsize, i : i + headsize] for i in range(0, self.to_add_out_columns, headsize)]
            )
            Hinv_diag_2 = torch.diagonal(torch.linalg.cholesky(Hinv_diag_2), dim1=-2, dim2=-1).reshape(-1)
            Hinv_diag_2 = Hinv_diag_2 ** 2

            error_1 = torch.sum(W1 ** 2 / Hinv_diag_1.unsqueeze(0), dim=0)
            error_2 = torch.sum(W2 ** 2 / Hinv_diag_2.unsqueeze(0), dim=0)
            error_1[self.column_mask] = torch.inf
            error_2[self.column_mask] = torch.inf

            head_sort_idx_1 = error_1.view(-1, headsize).sum(1).argsort()
            head_sort_idx_2 = error_2.view(-1, headsize).sum(1).argsort()
            head_merge_sort_idx = reciprocal_rank_fusion([head_sort_idx_1.tolist(), head_sort_idx_2.tolist()])
            sorted_heads = sorted(head_merge_sort_idx.keys(), key=lambda x: head_merge_sort_idx[x], reverse=True)
            column_sort_idx = torch.hstack(
                [torch.arange(head_idx * headsize, head_idx * headsize + headsize, device=self.dev) for head_idx in sorted_heads]
            )
            cnt = headsize

            W1 = W1[:, column_sort_idx]
            W2 = W2[:, column_sort_idx]
            Hinv_1 = Hinv_1[column_sort_idx, :][:, column_sort_idx]
            Hinv_2 = Hinv_2[column_sort_idx, :][:, column_sort_idx]
            Hinv_1 = torch.linalg.cholesky(Hinv_1, upper=True)[:cnt]
            Hinv_2 = torch.linalg.cholesky(Hinv_2, upper=True)[:cnt]

            W1_prune = W1[:, :cnt].clone()
            W2_prune = W2[:, :cnt].clone()
            local_hinv_1 = Hinv_1[:, :cnt]
            local_hinv_2 = Hinv_2[:, :cnt]
            Err1 = torch.zeros_like(W1_prune)
            Err2 = torch.zeros_like(W2_prune)

            for i in range(cnt):
                Err1[:, i : i + 1] = W1_prune[:, i : i + 1] / Hinv_1[i, i]
                Err2[:, i : i + 1] = W2_prune[:, i : i + 1] / Hinv_2[i, i]
                if not self.no_compensate:
                    W1_prune[:, i:] -= Err1[:, i : i + 1].matmul(local_hinv_1[i : i + 1, i:])
                    W2_prune[:, i:] -= Err2[:, i : i + 1].matmul(local_hinv_2[i : i + 1, i:])

            W1[:, :cnt] = 0
            W2[:, :cnt] = 0

            if not self.no_compensate:
                end = self.to_out_columns - pruned_columns
                W1[:, cnt:end] -= Err1.matmul(Hinv_1[:, cnt:end])
                W2[:, cnt:end] -= Err2.matmul(Hinv_2[:, cnt:end])

            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W1 = W1[:, column_sort_idx_inv]
            W2 = W2[:, column_sort_idx_inv]

            pruned_idx = column_sort_idx[:cnt]
            H1[pruned_idx, :] = H1[:, pruned_idx] = 0
            H1[pruned_idx, pruned_idx] = 1
            H2[pruned_idx, :] = H2[:, pruned_idx] = 0
            H2[pruned_idx, pruned_idx] = 1

            self.column_mask[pruned_idx] = True
            pruned_columns += cnt

        self.layer_to_out.weight.data = W1.reshape(self.layer_to_out.weight.shape).to(self.layer_to_out.weight.data.dtype)
        self.layer_to_add_out.weight.data = W2.reshape(self.layer_to_add_out.weight.shape).to(
            self.layer_to_add_out.weight.data.dtype
        )

        pruned_indices = torch.where(self.column_mask)[0]
        print(f"pruned columns {pruned_indices.numel()}/{self.to_out_columns} on joint attention")
        return pruned_indices

    def free(self):
        self.H_to_out = None
        self.H_to_add_out = None
        self.column_mask = None
        torch.cuda.empty_cache()
