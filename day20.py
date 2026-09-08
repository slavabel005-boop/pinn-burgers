import torch
import torch.nn as nn
import matplotlib.pyplot as plt

nu = 0.01 / torch.pi

torch.manual_seed(0)

xt_f = torch.stack([2 * torch.rand(1000) - 1, torch.rand(1000)], dim = 1)
xt_f.requires_grad_(True)

x_ic = torch.stack([torch.linspace(-1, 1, 100), torch.zeros(100)], dim = 1)

x_b = torch.cat([torch.full((50,), -1), torch.full((50,), 1)])
t_b = torch.linspace(0, 1, 100)
x_bc = torch.stack([x_b, t_b], dim = 1)

pinn = nn.Sequential(

    nn.Linear(2, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1)

)

optimizer = torch.optim.Adam(pinn.parameters())

for i in range(20000):

    optimizer.zero_grad()

    u = pinn(xt_f)

    grads = torch.autograd.grad(u, xt_f, torch.ones_like(u), create_graph = True)[0]

    du_dx = grads[:, 0:1]
    du_dt = grads[:, 1:2]

    d2u_dx2 = torch.autograd.grad(du_dx, xt_f, torch.ones_like(du_dx), create_graph = True)[0][:, 0:1]

    r = du_dt + u * du_dx - nu * d2u_dx2

    L_eq = (r ** 2).mean()

    L_ic = ((pinn(x_ic) + torch.sin(torch.pi * x_ic[:, 0:1])) ** 2).mean()

    L_bc = ((pinn(x_bc)) ** 2).mean()

    L = L_eq + L_ic + L_bc
    L.backward()

    optimizer.step()

print(" L = ", L.item())