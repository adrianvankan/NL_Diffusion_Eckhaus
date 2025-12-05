###################################################################################################################################
#### This code solves the nonlinear BVP arising from the similarity ansatz to the nonlinear diffusion equation (WKB limit of CGLE)
#### Author: Adrian van Kan;    Date: 12/05/2025
###################################################################################################################################


import numpy as np
from scipy.integrate import solve_bvp
import style

# ----------------------------------------
# Parameters
# ----------------------------------------
L = 20.0      # far-field cutoff (domain size)
N = 1000      # number of mesh points (initially, is adjusted dynamically)

# ----------------------------------------
# ODE system for y = [H, H', H'', H''']
# H'''' = -1/2 H + 1/4 \eta H' + (H'' H - (H')^2)/H^2
# ----------------------------------------
def fun(eta, y):
    H, Hp, Hpp, Hppp = y
    # avoid division by zero: small floor on H
    H_safe = np.where(H <= 1e-12, 1e-12, H)
    H4 = -0.5*H + 0.25*eta*Hp + (Hpp*H_safe - Hp**2) / H_safe**2
    return np.vstack((Hp, Hpp, Hppp, H4))

# ----------------------------------------
# Boundary conditions:
# at \eta = 0:  H'(0) = 0,  H'''(0) = 0  (even solution)
# at \eta = L:  H''(L) L^2 = 2 H(L),  H'(L) = L H''(L)  (parabolic tail)
# ----------------------------------------
def bc(ya, yb):
    H0, Hp0, Hpp0, Hppp0 = ya
    HL, HpL, HppL, HpppL = yb
    return np.array([
        Hp0,                     # H'(0) = 0
        Hppp0,                   # H'''(0) = 0
        HppL*L**2 - 2.0*HL,      # H''(L) L^2 = 2 H(L)
        HpL - L*HppL             # H'(L) = L H''(L)
    ])

# ----------------------------------------
# Initial mesh + initial guess
# ----------------------------------------
eta = np.linspace(0.0, L, N)

# crude initial guess: slightly parabolic, positive everywhere
a0 = 0.1
H_guess    = 1.0 + a0*eta**2
Hp_guess   = 2*a0*eta
Hpp_guess  = 2*a0*np.ones_like(eta)
Hppp_guess = np.zeros_like(eta)
y_guess = np.vstack((H_guess, Hp_guess, Hpp_guess, Hppp_guess))

# ----------------------------------------
# Solve BVP
# ----------------------------------------
sol = solve_bvp(fun, bc, eta, y_guess, tol=1e-4, max_nodes=10000)

print("BVP status:", sol.status, sol.message)

# ----------------------------------------
# Diagnostics and eigenvalue C
# ----------------------------------------
eta_fine = np.linspace(0.0, L, 400)
H, Hp, Hpp, Hppp = sol.sol(eta_fine)

HL, HpL, HppL, HpppL = sol.sol(L)
C_from_H   = HL / L**2
C_from_Hpp = HppL / 2.0

print("\nAt \\eta = L:")
print("H(L)      =", HL)
print("H'(L)     =", HpL,  "  (should be ~ L H''(L))")
print("H''(L)    =", HppL, "  (should give same C)")
print("Check shape conditions:")
print("H''(L) L^2 - 2 H(L) =", HppL*L**2 - 2.0*HL)
print("H'(L) - L H''(L)   =", HpL - L*HppL)

print("\nNonlinear eigenvalue C estimates:")
print("C = H(L)/L^2       =", C_from_H)
print("C = H''(L)/2       =", C_from_Hpp)

# Optional plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4),layout='constrained')
    plt.plot(eta_fine, H, label="$H(\\eta)$")
    plt.loglog(eta_fine, C_from_H * eta_fine**2, "k--",
             label="$C\\eta^2$")#label=fr"{C_from_H:.3g} $\eta^2$")
    plt.xlabel(r"$\eta$")
    plt.ylabel("$H(\eta)$")
    plt.legend()
    plt.grid(alpha=0.3)
    #plt.tight_layout()
    plt.ylim(0.01,0.0425)
    plt.savefig('H_vs_eta.pdf',dpi=200)
    plt.show()

