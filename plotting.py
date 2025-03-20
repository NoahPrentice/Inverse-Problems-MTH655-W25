import matplotlib.pyplot as plt
import numpy as np

def plot_solution(solution):
    A_values = np.linspace(0, 50, 2**10)
    c_values = solution.sol(A_values)[0]
    plt.plot(A_values, c_values)
    plt.xlabel("A")
    plt.ylabel("c")
    plt.title("True consumption of housing by assets")
    plt.show()


def plot_residual_history(residual_history, k):
    iterations = range(len(residual_history))
    plt.semilogy(iterations, residual_history, label="Residual Norm", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("LM Residual History (k = {})".format(k))
    plt.legend()
    plt.show()


def plot_stepsize_history(stepsize_history, k):
    iterations = range(len(stepsize_history))
    plt.semilogy(iterations, stepsize_history, label="Stepsize Norm", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Stepsize Norm")
    plt.title("LM Stepsize History (k = {})".format(k))
    plt.legend()
    plt.show()



def plot_error_convergence(error_history, k):
    iterations = range(len(error_history))
    plt.semilogy(iterations, error_history, label="Error Norm", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Error Norm")
    plt.title("Error Convergence (k = {})".format(k))
    plt.legend()
    plt.show()

def plot_data(A_values, data):
    plt.scatter(A_values, data, color="red", marker=",", s=1, label="Data")
    plt.xlabel("A")
    plt.ylabel("Data")
    plt.title("Data")
    plt.show()