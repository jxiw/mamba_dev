import torch
import causal_conv1d_cuda
from einops import rearrange

batch = 1
dim = 3584
seqlen = 2
width = 4
itype=torch.float32
device="cuda"

x = torch.ones(batch, dim, seqlen, device=device, dtype=itype).transpose(1, 2)

x = torch.nn.functional.pad(x, (0, 0, 1, 0), "constant", 0)[:, 1:, :].contiguous()

# x = torch.ones(batch, seqlen, dim, device=device, dtype=itype)
print(x.shape)
print(x.stride())

x = rearrange(x, "b s d -> b d s")
print(x.stride())

initial_states = torch.ones(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2)
conv1d_weight = torch.ones(dim, width, device=device, dtype=torch.float32)
conv1d_bias = torch.ones(dim, device=device, dtype=torch.float32)

conv_next_state = torch.ones(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2)

conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
    x, conv1d_weight, conv1d_bias, None, initial_states, conv_next_state, True
)

print("conv1d_out:", conv1d_out)