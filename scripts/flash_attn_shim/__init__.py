# Pure-torch stand-in for flash-attn on GPUs it has no kernels for: verl only imports the padding helpers below.
# Attention itself then runs through PyTorch SDPA (R1X4_ATTN_IMPL=sdpa, actor_rollout_ref.model.use_remove_padding=False).
__version__ = "0.0.0-shim"
