import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from plotting import *

np.random.seed(1)


# Problem setup
def ode_system(A, c, r, s):
    c1, c2 = c
    dc1dA = c2
    dc2dA = (s**2 / 2) * c2 / (c1 - r * A)
    return [dc1dA, dc2dA]


def simulate_ode(q, A_span=(0, 50), c0=[0.1, 0.1]):
    return solve_ivp(ode_system, A_span, c0, args=(q[0], q[1]), dense_output=True)


def build_residual(q):
    test_solution = simulate_ode(q)
    if use_noisy_data:
        return test_solution.sol(A_values)[0] - noisy_data
    return test_solution.sol(A_values)[0] - data


def build_jacobian(q, epsilon=1e-8):
    jacobian = np.zeros((len(A_values), len(q)))
    for i in range(len(q)):
        q_perturbed = q.copy()
        q_perturbed[i] += epsilon
        R_perturbed = build_residual(q_perturbed)
        R_original = build_residual(q)
        jacobian[:, i] = (R_perturbed - R_original) / epsilon
    return jacobian


# Optimization
def find_q_with_lm(initial_q, initial_nu=1, tol=1e-6, max_iterations=100):
    q = initial_q.copy()
    nu = initial_nu
    for _ in range(max_iterations):
        residual = build_residual(q)
        residual_norm = np.linalg.norm(residual)
        # If residual is small enough, we have converged
        if residual_norm < tol:
            print("Converged at iteration", _)
            break

        # Otherwise, we compute the Jacobian and Hessian and update q
        jacobian = build_jacobian(q)
        hessian = jacobian.T @ jacobian
        hessian += nu * np.eye(hessian.shape[0])
        gradient = jacobian.T @ residual
        step = np.linalg.solve(hessian, -gradient)
        q += step
        nu = initial_nu * np.linalg.norm(residual)
    return q


def find_residual_history_with_lm(q0, nu0=1, tol=1e-6, max_iterations=100):
    q = q0.copy()
    nu = nu0
    residual_history = []
    for _ in range(max_iterations):
        residual = build_residual(q)
        residual_norm = np.linalg.norm(residual)
        residual_history.append(residual_norm)
        if residual_norm < tol:
            print("Converged at iteration", _)
            break
        jacobian = build_jacobian(q)
        hessian = jacobian.T @ jacobian
        hessian += nu * np.eye(hessian.shape[0])
        gradient = jacobian.T @ residual
        step = np.linalg.solve(hessian, -gradient)
        q += step
        nu = nu0 * np.linalg.norm(residual)
    return residual_history


def find_q_history_with_lm(q0, nu0=1, tol=1e-6, max_iterations=100):
    q = q0.copy()
    nu = nu0
    q_history = []
    for _ in range(max_iterations):
        q_history.append(q.copy())
        residual = build_residual(q)
        residual_norm = np.linalg.norm(residual)
        if residual_norm < tol:
            print("Converged at iteration", _)
            break
        jacobian = build_jacobian(q)
        hessian = jacobian.T @ jacobian
        hessian += nu * np.eye(hessian.shape[0])
        gradient = jacobian.T @ residual
        step = np.linalg.solve(hessian, -gradient)
        q += step
        nu = nu0 * np.linalg.norm(residual)
    return q_history


def find_stepsize_history_with_lm(q0, nu0=1, tol=1e-6, max_iterations=100):
    q = q0.copy()
    nu = nu0
    stepsize_history = []
    for _ in range(max_iterations):
        residual = build_residual(q)
        residual_norm = np.linalg.norm(residual)
        if residual_norm < tol:
            print("Converged at iteration", _)
            break
        jacobian = build_jacobian(q)
        hessian = jacobian.T @ jacobian
        hessian += nu * np.eye(hessian.shape[0])
        gradient = jacobian.T @ residual
        step = np.linalg.solve(hessian, -gradient)
        stepsize_history.append(np.linalg.norm(step))
        q += step
        nu = nu0 * np.linalg.norm(residual)
    return stepsize_history


# Hypothesis testing
def build_solution_with_s_zero(r, A_values=np.linspace(0, 50, 2**10)):
    return r * A_values


def build_residual_with_s_zero(r):
    return build_solution_with_s_zero(r) - data


def find_r_with_s_zero(r0):
    return least_squares(build_residual_with_s_zero, r0).x


# Manufacturing the true solution
true_r = 0.1
true_s = 0.1
true_q = np.array([true_r, true_s])
solution = simulate_ode(true_q)
q0 = np.ones(2)

# Discretizing and observing data
k = 11
M = 2**k
A_values = np.linspace(0, 50, M)
data = solution.sol(A_values)[0]

use_noisy_data = True
noise_level = 0.1
noisy_data = data + noise_level * np.random.randn(M)


plot_solution(solution)
plot_data(A_values, noisy_data)
