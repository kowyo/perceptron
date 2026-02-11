import os
import numpy as np
import matplotlib.pyplot as plt

def plot_perceptron_learning(gate_name, X, y, eta=1.0, max_epochs=20):
    np.random.seed(42)
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    
    plt.figure(figsize=(6, 5))
    
    for i in range(len(y)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        plt.scatter(X[i, 0], X[i, 1] if X.shape[1]>1 else 0, 
                    c=color, marker=marker, s=100, zorder=5, label=f'Class {y[i]}' if i < 2 else "")

    lines_plotted = 0
    x_vals = np.linspace(-0.5, 1.5, 100)
    
    for epoch in range(max_epochs):
        converged = True
        for i in range(len(X)):
            if X.shape[1] == 2:
                if w[1] != 0:
                    y_vals = -(w[0] * x_vals + b) / w[1]
                    alpha = 0.2 + 0.8 * (lines_plotted / 15)
                    plt.plot(x_vals, y_vals, color='black', alpha=min(alpha, 0.6), linestyle='--')
                    lines_plotted += 1
            else:
                plt.axvline(x=-b/w[0], color='black', alpha=0.3, linestyle='--')
                lines_plotted += 1

            linear_output = np.dot(X[i], w) + b
            y_pred = 1 if linear_output >= 0 else 0
            error = y[i] - y_pred
            if error != 0:
                w += eta * error * X[i]
                b += eta * error
                converged = False
        if converged: break

    if X.shape[1] == 2 and w[1] != 0:
        plt.plot(x_vals, -(w[0] * x_vals + b) / w[1], color='green', linewidth=3, label='Final Boundary')
    elif X.shape[1] == 1:
        plt.axvline(x=-b/w[0], color='green', linewidth=3, label='Final Boundary')

    plt.title(f"Perceptron Learning Trajectory: {gate_name}")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$" if X.shape[1]>1 else "N/A")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{gate_name}.png"))
    plt.close()

data = {
    "AND": (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,0,0,1])),
    "OR": (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,1])),
    "NAND": (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([1,1,1,0])),
    "COMPLEMENT": (np.array([[0], [1]]), np.array([1,0])),
    "XOR": (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,0]))
}

for name, (X, y) in data.items():
    plot_perceptron_learning(name, X, y)
