"""flash_attn.bert_padding re-implemented in pure torch (same API as flash-attn 2.x)."""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat  # noqa: F401  (verl imports rearrange from here)


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        flat = rearrange(input, "b ... -> b (...)")
        return torch.gather(flat, 0, repeat(indices, "z -> z d", d=flat.shape[1])).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        g = rearrange(grad_output, "b ... -> b (...)")
        out = torch.zeros(ctx.first_axis_dim, g.shape[1], device=g.device, dtype=g.dtype)
        out.scatter_(0, repeat(indices, "z -> z d", d=g.shape[1]), g)
        return out.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        out = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        out[indices] = values
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    seqlens = all_masks.sum(dim=-1, dtype=torch.int32)
    used = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices), indices, cu, int(seqlens.max()), used


def pad_input(hidden_states, indices, batch, seqlen):
    return rearrange(index_put_first_axis(hidden_states, indices, batch * seqlen), "(b s) ... -> b s ...", b=batch)
