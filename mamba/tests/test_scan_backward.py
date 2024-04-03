
# export PATH=/home/jw2544/miniconda3/envs/mamba_dev/bin/:$PATH
# export MAX_JOBS=32

import torch
import selective_scan_cuda
from torch.cuda.amp import custom_bwd, custom_fwd

batch_size = 2
dim = 2048
dstate = 16
seqlen = 4096*2+300
device = 'cuda'
wtype = torch.float32
itype = torch.float32

# u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
# delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
# A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype))
# B_shape = (batch_size, 1, dstate, seqlen)
# B = torch.randn(*B_shape, device=device, dtype=wtype)
# C_shape = (batch_size, 1, dstate, seqlen)
# C = torch.randn(*C_shape, device=device, dtype=wtype)
# D = torch.randn(dim, device=device, dtype=torch.float32)
# z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype) 
# delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32))

# delta_softplus=True
# out_ref, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, None)

# out_ref_first, out_ref_second = torch.chunk(out_ref, chunks=2, dim=-1)

# last_state = x[:, :, -1, 1::2]

# B_first, B_second = torch.chunk(B, chunks=2, dim=-1)
# C_first, C_second = torch.chunk(C, chunks=2, dim=-1)
# z_first, z_second = torch.chunk(z, chunks=2, dim=-1)
# u_first, u_second = torch.chunk(u, chunks=2, dim=-1)
# delta_first, delta_second = torch.chunk(delta, chunks=2, dim=-1)

# resume_state = torch.zeros((batch_size, dim, dstate)).cuda()

# out_first, x_first, *rest = selective_scan_cuda.fwd(u_first, delta_first, A, B_first, C_first, D, z_first, delta_bias, delta_softplus, None)
# first_last_state = x_first[:, :, -1, 1::2]

# print("resume_state:", resume_state.stride())
# print("first_last_state:", first_last_state.shape)

# resume_state.copy_(x_first[:, :, -1, 1::2])

# print(resume_state.shape)
# print(resume_state.stride())

# print("======================")
# out_second, x_second, *rest = selective_scan_cuda.fwd(u_second, delta_second, A, B_second, C_second, D, z_second, delta_bias, delta_softplus, resume_state)
# second_last_state = x_second[:, :, -1, 1::2]

rtol, atol = (6e-4, 2e-3)

# print(f'Output max diff: {(out_ref_first - out_first).abs().max().item()}')
# print(f'Output mean diff: {(out_ref_first - out_first).abs().mean().item()}')
# assert torch.allclose(out_first, out_ref_first, rtol=rtol, atol=atol)

# print(f'Output max diff: {(out_ref_second - out_second).abs().max().item()}')
# print(f'Output mean diff: {(out_ref_second - out_second).abs().mean().item()}')
# assert torch.allclose(out_second, out_ref_second, rtol=rtol, atol=atol)

# print(f'Output state diff: {(last_state - second_last_state).abs().max().item()}')
# assert torch.allclose(last_state, second_last_state, rtol=rtol, atol=atol)

print("======test backward grad======")

from torch.autograd import Function, gradcheck

class CustomFunction(Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx, u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, delta_softplus, ssm_state):
        out_ref, scan_intermediates, out_z = selective_scan_cuda.fwd(u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, True, ssm_state)
        ctx.save_for_backward(u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, out_ref, scan_intermediates, ssm_state)
        last_state = scan_intermediates[:, :, -1, 1::2]
        return out_ref, last_state, out_z

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, last_state, out_z):
        (u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, out_ref, scan_intermediates, ssm_state) = ctx.saved_tensors
        dz = torch.empty_like(z_ref)
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, dout, scan_intermediates, out_ref, None, True,
            False,
            ssm_state,
        )
        dz = rest[0]
        return (du, ddelta, dA, dB, dC,
                dD,
                dz,
                ddelta_bias,
                None,
                None)

def customer_fn(u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, delta_softplus, ssm_state):
    return CustomFunction.apply(u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, delta_softplus, ssm_state)

u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype, requires_grad=True))
B_shape = (batch_size, 1, dstate, seqlen)
B = torch.randn(*B_shape, device=device, dtype=wtype, requires_grad=True)
C_shape = (batch_size, 1, dstate, seqlen)
C = torch.randn(*C_shape, device=device, dtype=wtype, requires_grad=True)
D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True) 
delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32, requires_grad=True))
delta_softplus=True

A_ref = A.detach().clone().requires_grad_()
B_ref = B.detach().clone().requires_grad_()
C_ref = C.detach().clone().requires_grad_()
D_ref = D.detach().clone().requires_grad_()
z_ref = z.detach().clone().requires_grad_()
u_ref = u.detach().clone().requires_grad_()
delta_ref = delta.detach().clone().requires_grad_()
delta_bias_ref = delta_bias.detach().clone().requires_grad_()
    
out_ref, last_state, *rest = customer_fn(u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, delta_softplus, None)

out_ref_first, out_ref_second = torch.chunk(out_ref, chunks=2, dim=-1)

B_first, B_second = torch.chunk(B, chunks=2, dim=-1)
C_first, C_second = torch.chunk(C, chunks=2, dim=-1)
z_first, z_second = torch.chunk(z, chunks=2, dim=-1)
u_first, u_second = torch.chunk(u, chunks=2, dim=-1)
delta_first, delta_second = torch.chunk(delta, chunks=2, dim=-1)

resume_state = torch.zeros((batch_size, dim, dstate)).cuda()

A_first = A.detach().clone().requires_grad_()
out_first, x_first, *rest = customer_fn(u_first, delta_first, A_first, B_first, C_first, D, z_first, delta_bias, delta_softplus, None)

resume_state.copy_(x_first)

B_second = B_second.detach().clone().requires_grad_()
C_second = C_second.detach().clone().requires_grad_()
z_second = z_second.detach().clone().requires_grad_()
u_second = u_second.detach().clone().requires_grad_()
delta_second = delta_second.detach().clone().requires_grad_()

A_second = A.detach().clone().requires_grad_()
D_second = D.detach().clone().requires_grad_()
delta_bias_second = delta_bias.detach().clone().requires_grad_()

out_second, x_second, *rest = customer_fn(u_second, delta_second, A_second, B_second, C_second, D_second, z_second, delta_bias_second, delta_softplus, resume_state)

g = torch.randn_like(out_ref)
out_ref.backward(g)

g_first, g_second = torch.chunk(g, chunks=2, dim=-1)

out_first.backward(g_first, retain_graph=True)
out_second.backward(g_second, retain_graph=True)

u_ref_grad = u_ref.grad[:, :, u_ref.shape[2]//2:]
delta_ref_grad = u_ref.grad[:, :, delta_ref.shape[2]//2:]
B_ref_grad = B_ref.grad[:, :, :, B_ref.shape[3]//2:]
C_ref_grad = C_ref.grad[:, :, :, C_ref.grad.shape[3]//2:]

# print("z_ref_grad.grad:", z_ref.grad.shape)
z_ref_grad = z_ref.grad[:, :, z_ref.shape[2]//2:]

print("A_first.shape:", A_first.grad.shape)
print("A_second.shape:", A_second.grad.shape)

print(f'du max diff: {(u_second.grad - u_ref_grad).abs().max().item()}')
print(f'ddelta max diff: {(delta_second.grad - delta_ref_grad).abs().max().item()}')
print(f'dA max diff: {(A_first.grad * A_second.grad - A_ref.grad).abs().max().item()}')
print(f'dB max diff: {(B_second.grad - B_ref_grad).abs().max().item()}')
print(f'dC max diff: {(C_second.grad - C_ref_grad).abs().max().item()}')
print(f'dD max diff: {(D_second.grad - D_ref.grad).abs().max().item()}')
print(f'dz max diff: {(z_second.grad - z_ref_grad).abs().max().item()}')
print(f'ddelta_bias max diff: {(delta_bias_second.grad - delta_bias_ref.grad).abs().max().item()}')

# assert torch.allclose(u_second.grad, u_ref_grad.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
# assert torch.allclose(delta_second.grad, delta_ref_grad.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
# assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
# assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
#                         atol=atolw if not is_variable_B else atol)
# assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
#                         atol=atolw if not is_variable_C else atol)
# assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
# assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
# assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)


