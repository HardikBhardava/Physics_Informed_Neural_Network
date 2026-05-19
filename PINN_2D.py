import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

torch.manual_seed(0)

# ============================================================
# Parameters
# ============================================================
alpha = 0.01
a, b = 0.1, 0.3

lam_pde = 200.0
lam_adj = 200.0
lam_opt = 500.0
lam_cost = 1.0


# ============================================================
# Neural network
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=8):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


u_raw = MLP().to(device)   # state
z_raw = MLP().to(device)   # adjoint
q_raw = MLP().to(device)   # control


# ============================================================
# Hard boundary conditions
# u = z = 0 on boundary of (0,1)^2
# ============================================================
def boundary_factor(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return x * (1 - x) * y * (1 - y)


def u_net(X):
    return boundary_factor(X) * u_raw(X)


def z_net(X):
    return boundary_factor(X) * z_raw(X)


def q_net(X):
    return a + (b - a) * torch.sigmoid(q_raw(X))


# ============================================================
# Exact manufactured solution
# ============================================================
def exact_u(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return x * (1 - x) * y * (1 - y)


def exact_z(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.05 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def exact_q(X):
    return torch.clamp((1.0 / alpha) * exact_u(X) * exact_z(X), a, b)


def qd_fun(X):
    return torch.zeros_like(X[:, 0:1])


# State PDE:
# -Delta u + q u = f
def f_fun(X):
    x = X[:, 0:1]
    y = X[:, 1:2]

    u = exact_u(X)
    q = exact_q(X)

    minus_delta_u = 2 * y * (1 - y) + 2 * x * (1 - x)

    return minus_delta_u + q * u


# Adjoint PDE:
# -Delta z + q z = u - u_d
# so u_d = u - (-Delta z + q z)
def ud_fun(X):
    u = exact_u(X)
    z = exact_z(X)
    q = exact_q(X)

    x = X[:, 0:1]
    y = X[:, 1:2]

    minus_delta_z = (
        0.1 * math.pi**2
        * torch.sin(math.pi * x)
        * torch.sin(math.pi * y)
    )

    return u - (minus_delta_z + q * z)


# ============================================================
# Laplacian using autograd
# ============================================================
def laplacian(y, X):
    grad_y = torch.autograd.grad(
        y, X,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

    y_x = grad_y[:, 0:1]
    y_y = grad_y[:, 1:2]

    y_xx = torch.autograd.grad(
        y_x, X,
        grad_outputs=torch.ones_like(y_x),
        create_graph=True
    )[0][:, 0:1]

    y_yy = torch.autograd.grad(
        y_y, X,
        grad_outputs=torch.ones_like(y_y),
        create_graph=True
    )[0][:, 1:2]

    return y_xx + y_yy


# ============================================================
# Optimizer
# ============================================================
params = (
    list(u_raw.parameters())
    + list(z_raw.parameters())
    + list(q_raw.parameters())
)

optimizer = torch.optim.Adam(params, lr=1e-3)


# ============================================================
# Adam training
# ============================================================
for epoch in range(5000):

    X = torch.rand(500, 2, device=device)
    X.requires_grad_(True)

    u = u_net(X)
    z = z_net(X)
    q = q_net(X)

    delta_u = laplacian(u, X)
    delta_z = laplacian(z, X)

    # State PDE: -Delta u + q u = f
    pde_res = -delta_u + q * u - f_fun(X)

    # Adjoint PDE: -Delta z + q z = u - u_d
    adj_res = -delta_z + q * z - (u - ud_fun(X))

    # Projection optimality condition:
    # q = P_[a,b](1/alpha * u*z + q_d)
    q_proj = torch.clamp((1.0 / alpha) * u * z + qd_fun(X), a, b)
    opt_res = q - q_proj

    loss_pde = torch.mean(pde_res**2)
    loss_adj = torch.mean(adj_res**2)
    loss_opt = torch.mean(opt_res**2)

    loss_track = 0.5 * torch.mean((u - ud_fun(X))**2)
    loss_control = 0.5 * alpha * torch.mean((q - qd_fun(X))**2)

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
            X_test = torch.rand(1000, 2, device=device)

            err_u = torch.sqrt(torch.mean((u_net(X_test) - exact_u(X_test))**2))
            err_z = torch.sqrt(torch.mean((z_net(X_test) - exact_z(X_test))**2))
            err_q = torch.sqrt(torch.mean((q_net(X_test) - exact_q(X_test))**2))

        print(
            f"Epoch {epoch:5d} | "
            f"loss={loss.item():.4e} | "
            f"pde={loss_pde.item():.4e} | "
            f"adj={loss_adj.item():.4e} | "
            f"opt={loss_opt.item():.4e} | "
            f"err_u={err_u.item():.4e} | "
            f"err_z={err_z.item():.4e} | "
            f"err_q={err_q.item():.4e}"
        )


# ============================================================
# LBFGS refinement
# ============================================================
N_lbf = 50
x = torch.linspace(0, 1, N_lbf, device=device)
y = torch.linspace(0, 1, N_lbf, device=device)
XX, YY = torch.meshgrid(x, y, indexing="ij")

X_lbfgs = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)
X_lbfgs.requires_grad_(True)

optimizer_lbfgs = torch.optim.LBFGS(
    params,
    lr=1.0,
    max_iter=1500,
    max_eval=1500,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe"
)


def closure():
    optimizer_lbfgs.zero_grad()

    u = u_net(X_lbfgs)
    z = z_net(X_lbfgs)
    q = q_net(X_lbfgs)

    delta_u = laplacian(u, X_lbfgs)
    delta_z = laplacian(z, X_lbfgs)

    pde_res = -delta_u + q * u - f_fun(X_lbfgs)
    adj_res = -delta_z + q * z - (u - ud_fun(X_lbfgs))

    q_proj = torch.clamp((1.0 / alpha) * u * z + qd_fun(X_lbfgs), a, b)
    opt_res = q - q_proj

    loss_pde = torch.mean(pde_res**2)
    loss_adj = torch.mean(adj_res**2)
    loss_opt = torch.mean(opt_res**2)

    loss_track = 0.5 * torch.mean((u - ud_fun(X_lbfgs))**2)
    loss_control = 0.5 * alpha * torch.mean((q - qd_fun(X_lbfgs))**2)

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


# ============================================================
# Final errors
# ============================================================
with torch.no_grad():
    X_test = torch.rand(5000, 2, device=device)

    err_u = torch.sqrt(torch.mean((u_net(X_test) - exact_u(X_test))**2))
    err_z = torch.sqrt(torch.mean((z_net(X_test) - exact_z(X_test))**2))
    err_q = torch.sqrt(torch.mean((q_net(X_test) - exact_q(X_test))**2))

print("\nFinal errors")
print("============")
print("L2-like error u:", err_u.item())
print("L2-like error z:", err_z.item())
print("L2-like error q:", err_q.item())


# ============================================================
# 3D wireframe plots
# ============================================================
N_plot = 80

x = torch.linspace(0, 1, N_plot, device=device)
y = torch.linspace(0, 1, N_plot, device=device)
XX, YY = torch.meshgrid(x, y, indexing="ij")

X_plot = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)

with torch.no_grad():
    U_pred = u_net(X_plot).reshape(N_plot, N_plot).cpu().numpy()
    Z_pred = z_net(X_plot).reshape(N_plot, N_plot).cpu().numpy()
    Q_pred = q_net(X_plot).reshape(N_plot, N_plot).cpu().numpy()

    U_true = exact_u(X_plot).reshape(N_plot, N_plot).cpu().numpy()
    Z_true = exact_z(X_plot).reshape(N_plot, N_plot).cpu().numpy()
    Q_true = exact_q(X_plot).reshape(N_plot, N_plot).cpu().numpy()

XX_np = XX.cpu().numpy()
YY_np = YY.cpu().numpy()


# def wireframe_compare(XX, YY, true_val, pred_val, title, zlabel):
#     fig = plt.figure(figsize=(13, 5))

#     ax1 = fig.add_subplot(1, 2, 1, projection="3d")
#     ax1.plot_wireframe(XX, YY, true_val, rstride=3, cstride=3)
#     ax1.set_title("True " + title)
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("y")
#     ax1.set_zlabel(zlabel)

#     ax2 = fig.add_subplot(1, 2, 2, projection="3d")
#     ax2.plot_wireframe(XX, YY, pred_val, rstride=3, cstride=3)
#     ax2.set_title("Predicted " + title)
#     ax2.set_xlabel("x")
#     ax2.set_ylabel("y")
#     ax2.set_zlabel(zlabel)

#     plt.tight_layout()
#     plt.show()


# wireframe_compare(XX_np, YY_np, U_true, U_pred, "state u(x,y)", "u")
# wireframe_compare(XX_np, YY_np, Q_true, Q_pred, "control q(x,y)", "q")
# wireframe_compare(XX_np, YY_np, Z_true, Z_pred, "adjoint z(x,y)", "z")


# # ============================================================
# # Error surface plots
# # ============================================================
# wireframe_compare(
#     XX_np,
#     YY_np,
#     np.zeros_like(U_true),
#     np.abs(U_true - U_pred),
#     "absolute error |u_true - u_pred|",
#     "error"
# )

# wireframe_compare(
#     XX_np,
#     YY_np,
#     np.zeros_like(Q_true),
#     np.abs(Q_true - Q_pred),
#     "absolute error |q_true - q_pred|",
#     "error"
# )

# wireframe_compare(
#     XX_np,
#     YY_np,
#     np.zeros_like(Z_true),
#     np.abs(Z_true - Z_pred),
#     "absolute error |z_true - z_pred|",
#     "error"
# )


from matplotlib.lines import Line2D

def wireframe_true_pred(XX, YY, true_val, pred_val, title, zlabel):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_wireframe(
        XX, YY, true_val,
        rstride=3, cstride=3,
        color="blue",
        linewidth=0.8,
        label="true"
    )

    ax.plot_wireframe(
        XX, YY, pred_val,
        rstride=3, cstride=3,
        color="red",
        linewidth=0.8,
        linestyle="--",
        label="predicted"
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(zlabel)

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="true"),
        Line2D([0], [0], color="red", lw=2, linestyle="--", label="predicted"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

wireframe_true_pred( XX_np, YY_np, U_true, U_pred, "State: true vs predicted", "u(x,y)")

wireframe_true_pred( XX_np, YY_np, Z_true, Z_pred, "Adjoint: true vs predicted", "z(x,y)")

wireframe_true_pred(XX_np, YY_np, Q_true, Q_pred, "Control: true vs predicted", "q(x,y)")

def plot_control_active_regions(XX, YY, Q_true, Q_pred, a=0.1, b=0.3, tol=1e-3):
    fig = plt.figure(figsize=(14, 6))

    # True control regions
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_wireframe(
        XX, YY, Q_true,
        rstride=3, cstride=3,
        color="blue",
        linewidth=0.8
    )

    active_lower_true = np.abs(Q_true - a) < tol
    active_upper_true = np.abs(Q_true - b) < tol
    inactive_true = (Q_true > a + tol) & (Q_true < b - tol)

    ax1.scatter(
        XX[inactive_true], YY[inactive_true], Q_true[inactive_true],
        color="green", s=6, label="inactive: a < q < b"
    )
    ax1.scatter(
        XX[active_lower_true], YY[active_lower_true], Q_true[active_lower_true],
        color="orange", s=6, label="active lower: q = a"
    )
    ax1.scatter(
        XX[active_upper_true], YY[active_upper_true], Q_true[active_upper_true],
        color="purple", s=6, label="active upper: q = b"
    )

    ax1.set_title("True control active/inactive regions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("q(x,y)")
    ax1.legend()

    # Predicted control regions
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_wireframe(
        XX, YY, Q_pred,
        rstride=3, cstride=3,
        color="red",
        linewidth=0.8
    )

    active_lower_pred = np.abs(Q_pred - a) < tol
    active_upper_pred = np.abs(Q_pred - b) < tol
    inactive_pred = (Q_pred > a + tol) & (Q_pred < b - tol)

    ax2.scatter(
        XX[inactive_pred], YY[inactive_pred], Q_pred[inactive_pred],
        color="green", s=6, label="inactive: a < q < b"
    )
    ax2.scatter(
        XX[active_lower_pred], YY[active_lower_pred], Q_pred[active_lower_pred],
        color="orange", s=6, label="active lower: q = a"
    )
    ax2.scatter(
        XX[active_upper_pred], YY[active_upper_pred], Q_pred[active_upper_pred],
        color="purple", s=6, label="active upper: q = b"
    )

    ax2.set_title("Predicted control active/inactive regions")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("q(x,y)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_control_active_regions(
    XX_np, YY_np,
    Q_true, Q_pred,
    a=a,
    b=b,
    tol=1e-3
)

def plot_active_region_topview(XX, YY, Q_true, Q_pred, a=0.1, b=0.3, tol=1e-3):
    true_region = np.zeros_like(Q_true)
    pred_region = np.zeros_like(Q_pred)

    # 0 = inactive, 1 = lower active, 2 = upper active
    true_region[np.abs(Q_true - a) < tol] = 1
    true_region[np.abs(Q_true - b) < tol] = 2

    pred_region[np.abs(Q_pred - a) < tol] = 1
    pred_region[np.abs(Q_pred - b) < tol] = 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c1 = axes[0].contourf(XX, YY, true_region, levels=[-0.5, 0.5, 1.5, 2.5])
    axes[0].set_title("True active/inactive regions")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    c2 = axes[1].contourf(XX, YY, pred_region, levels=[-0.5, 0.5, 1.5, 2.5])
    axes[1].set_title("Predicted active/inactive regions")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()

plot_active_region_topview(
    XX_np, YY_np,
    Q_true, Q_pred,
    a=a,
    b=b,
    tol=1e-3
)