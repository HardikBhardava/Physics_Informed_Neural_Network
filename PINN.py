import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Parameters
alpha = 0.01
a, b = 0.1, 0.3

lam_pde = 1000.0
lam_adj = 1000.0
lam_opt = 1000.0
lam_cost = 1.0

# Neural network
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

u_raw = MLP().to(device)
z_raw = MLP().to(device)
q_raw = MLP().to(device)

# Hard boundary conditions
def u_net(x):
    return x * (1 - x) * u_raw(x)

def z_net(x):
    return x * (1 - x) * z_raw(x)

# Box constrained control
def q_net(x):
    return a + (b - a) * torch.sigmoid(q_raw(x))

# Exact manufactured solution
def exact_u(x):
    return x * (1 - x)

def exact_z(x):
    return 0.05 * torch.sin(math.pi * x)

def exact_q(x):
    return torch.clamp(5.0 * torch.sin(math.pi * x) * x * (1 - x), a, b)

def qd_fun(x):
    return torch.zeros_like(x)

def f_fun(x):
    u = exact_u(x)
    q = exact_q(x)
    return 2.0 + q * u

def ud_fun(x):
    u = exact_u(x)
    z = exact_z(x)
    q = exact_q(x)

    minus_z_xx = 0.05 * math.pi**2 * torch.sin(math.pi * x)

    return u - (minus_z_xx + q * z)

def second_derivative(y, x):
    y_x = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

    y_xx = torch.autograd.grad(
        y_x, x,
        grad_outputs=torch.ones_like(y_x),
        create_graph=True
    )[0]

    return y_xx

params = (
    list(u_raw.parameters())
    + list(z_raw.parameters())
    + list(q_raw.parameters())
)

optimizer = torch.optim.Adam(params, lr=1e-3)

# Training with Adam
for epoch in range(30000):
    x = torch.rand(1500, 1, device=device)
    x.requires_grad_(True)

    u = u_net(x)
    z = z_net(x)
    q = q_net(x)

    u_xx = second_derivative(u, x)
    z_xx = second_derivative(z, x)

    # State PDE: -u'' + q u = f
    pde_res = -u_xx + q * u - f_fun(x)

    # Adjoint PDE: -z'' + q z = u - u_d
    adj_res = -z_xx + q * z - (u - ud_fun(x))

    # Optimality condition:
    # q = P_[a,b](1/alpha * u*z + q_d)
    q_proj = torch.clamp((1.0 / alpha) * u * z + qd_fun(x), a, b)
    opt_res = q - q_proj

    # Cost terms
    loss_track = 0.5 * torch.mean((u - ud_fun(x))**2)
    loss_control = 0.5 * alpha * torch.mean((q - qd_fun(x))**2)

    loss_pde = torch.mean(pde_res**2)
    loss_adj = torch.mean(adj_res**2)
    loss_opt = torch.mean(opt_res**2)

    loss = (
        lam_pde * loss_pde
        + lam_adj * loss_adj
        + lam_opt * loss_opt
        + lam_cost * (loss_track + loss_control)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 3000 == 0:
        with torch.no_grad():
            x_test = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
            err_u = torch.sqrt(torch.mean((u_net(x_test) - exact_u(x_test))**2))
            err_q = torch.sqrt(torch.mean((q_net(x_test) - exact_q(x_test))**2))
            err_z = torch.sqrt(torch.mean((z_net(x_test) - exact_z(x_test))**2))

        print(
            epoch,
            "loss:", loss.item(),
            "pde:", loss_pde.item(),
            "adj:", loss_adj.item(),
            "opt:", loss_opt.item(),
            "err_u:", err_u.item(),
            "err_q:", err_q.item(),
            "err_z:", err_z.item()
        )

# Optional LBFGS refinement
optimizer_lbfgs = torch.optim.LBFGS(
    params,
    lr=1.0,
    max_iter=3000,
    max_eval=3000,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe"
)

x_lbfgs = torch.linspace(0, 1, 2000, device=device).reshape(-1, 1)
x_lbfgs.requires_grad_(True)

def closure():
    optimizer_lbfgs.zero_grad()

    u = u_net(x_lbfgs)
    z = z_net(x_lbfgs)
    q = q_net(x_lbfgs)

    u_xx = second_derivative(u, x_lbfgs)
    z_xx = second_derivative(z, x_lbfgs)

    pde_res = -u_xx + q * u - f_fun(x_lbfgs)
    adj_res = -z_xx + q * z - (u - ud_fun(x_lbfgs))

    q_proj = torch.clamp((1.0 / alpha) * u * z + qd_fun(x_lbfgs), a, b)
    opt_res = q - q_proj

    loss_pde = torch.mean(pde_res**2)
    loss_adj = torch.mean(adj_res**2)
    loss_opt = torch.mean(opt_res**2)

    loss_track = 0.5 * torch.mean((u - ud_fun(x_lbfgs))**2)
    loss_control = 0.5 * alpha * torch.mean((q - qd_fun(x_lbfgs))**2)

    loss = (
        lam_pde * loss_pde
        + lam_adj * loss_adj
        + lam_opt * loss_opt
        + lam_cost * (loss_track + loss_control)
    )

    loss.backward()
    return loss

print("Starting LBFGS refinement...")
optimizer_lbfgs.step(closure)

# Plot results
x_plot = torch.linspace(0, 1, 500, device=device).reshape(-1, 1)

with torch.no_grad():
    u_pred = u_net(x_plot).cpu().numpy()
    z_pred = z_net(x_plot).cpu().numpy()
    q_pred = q_net(x_plot).cpu().numpy()

    u_true = exact_u(x_plot).cpu().numpy()
    z_true = exact_z(x_plot).cpu().numpy()
    q_true = exact_q(x_plot).cpu().numpy()

x_np = x_plot.cpu().numpy()

plt.figure(figsize=(7, 4))
plt.plot(x_np, u_true, label="true u")
plt.plot(x_np, u_pred, "--", label="predicted u")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Optimal state")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(x_np, q_true, label="true q")
plt.plot(x_np, q_pred, "--", label="predicted q")
plt.xlabel("x")
plt.ylabel("q(x)")
plt.title("Optimal control")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(x_np, z_true, label="true z")
plt.plot(x_np, z_pred, "--", label="predicted z")
plt.xlabel("x")
plt.ylabel("z(x)")
plt.title("Adjoint state")
plt.legend()
plt.grid(True)
plt.show()