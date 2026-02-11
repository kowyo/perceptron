import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))

def plot_and_trajectories(etas):
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_range = np.linspace(-0.5, 1.5, 100)

    for ax, eta in zip(axes, etas):
        w = np.random.randn(2)
        b = np.random.randn()
        
        for i in range(len(y)):
            ax.scatter(X[i,0], X[i,1], c='blue' if y[i]==1 else 'red', 
                       marker='o' if y[i]==1 else 'x', s=100, zorder=10)

        update_count = 0
        max_iter = 100
        
        if w[1] != 0:
            ax.plot(x_range, -(w[0]*x_range + b)/w[1], 'k--', alpha=0.2, label='Initial')

        for epoch in range(max_iter):
            converged = True
            for i in range(len(X)):
                linear_output = np.dot(X[i], w) + b
                y_pred = 1 if linear_output >= 0 else 0
                
                if y_pred != y[i]:
                    error = y[i] - y_pred
                    w = w + eta * error * X[i]
                    b = b + eta * error
                    update_count += 1
                    converged = False
                    
                    if w[1] != 0:
                        y_line = -(w[0]*x_range + b)/w[1]
                        ax.plot(x_range, y_line, 'k-', alpha=0.15)
            
            if converged: break

        if w[1] != 0:
            final_y = -(w[0]*x_range + b)/w[1]
            ax.plot(x_range, final_y, 'g-', linewidth=3, label='Final Boundary')

        ax.set_title(f"Learning Rate $\eta$ = {eta}\nTotal Updates: {update_count}")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(fname="lr.png")
    plt.show()

plot_and_trajectories([0.01, 1.0, 10.0])