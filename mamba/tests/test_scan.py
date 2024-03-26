
# export PATH=/home/jw2544/miniconda3/envs/mamba_dev/bin/:$PATH
# export MAX_JOBS=32

import torch
import selective_scan_cuda

batch_size = 2
dim = 2048
dstate = 16
seqlen = 4096
device = 'cuda'
wtype = torch.float32
itype = torch.float32

u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype))
B_shape = (batch_size, 1, dstate, seqlen)
B = torch.randn(*B_shape, device=device, dtype=wtype)
C_shape = (batch_size, 1, dstate, seqlen)
C = torch.randn(*C_shape, device=device, dtype=wtype)
D = torch.randn(dim, device=device, dtype=torch.float32)
z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype) 
delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32))

delta_softplus=True
out_ref, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, None)

out_ref_first, out_ref_second = torch.chunk(out_ref, chunks=2, dim=-1)

# print("A.shape:", A.shape)
# print("B.shape:", B.shape)
# print("C.shape:", C.shape)
# print("D.shape:", D.shape)
# print("z.shape:", z.shape)
# print("u.shape:", u.shape)
# print("delta_bias.shape:", delta_bias.shape)
# print("delta.shape:", delta.shape)

# print("x.shape:", x.shape) # [2, 4, 2, 16]
# [bsz, hidden size, 2, state size]

last_state = x[:, :, -1, 1::2]

# print(last_state.shape) # [2, 4, 8]

B_first, B_second = torch.chunk(B, chunks=2, dim=-1)
C_first, C_second = torch.chunk(C, chunks=2, dim=-1)
z_first, z_second = torch.chunk(z, chunks=2, dim=-1)
u_first, u_second = torch.chunk(u, chunks=2, dim=-1)
delta_first, delta_second = torch.chunk(delta, chunks=2, dim=-1)

out_first, x_first, *rest = selective_scan_cuda.fwd(u_first, delta_first, A, B_first, C_first, D, z_first, delta_bias, delta_softplus, None)
first_last_state = x_first[:, :, -1, 1::2]

# print(x_first)
# print("x_first.shape:", x_first.shape)

x_first[:, :, 0, :] = x_first[:, :, -1, :]
x_first[:, :, 1:, :] = 0

# print(x_first.shape)

out_second, x_second, *rest = selective_scan_cuda.fwd(u_second, delta_second, A, B_second, C_second, D, z_second, delta_bias, delta_softplus, x_first)
second_last_state = x_second[:, :, -1, 1::2]

rtol, atol = (6e-4, 2e-3)

print(f'Output max diff: {(out_ref_first - out_first).abs().max().item()}')
print(f'Output mean diff: {(out_ref_first - out_first).abs().mean().item()}')
assert torch.allclose(out_first, out_ref_first, rtol=rtol, atol=atol)

print(f'Output max diff: {(out_ref_second - out_second).abs().max().item()}')
print(f'Output mean diff: {(out_ref_second - out_second).abs().mean().item()}')
assert torch.allclose(out_second, out_ref_second, rtol=rtol, atol=atol)

print(f'Output state diff: {(last_state - second_last_state).abs().max().item()}')
assert torch.allclose(last_state, second_last_state, rtol=rtol, atol=atol)

# print("out_second:", out_second)
# print("out_ref_second:", out_ref_second)

# print("last_state:", last_state)
# print("second_last_state:", second_last_state)
