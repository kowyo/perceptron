import numpy as np
import matplotlib.pyplot as plt

X = np.array([0, 0.9, 1.7, 3.1, 4.1, 5.1])
Y = np.array([0.6, 1.1, 3.9, 5.1, 6.2, 7.9])

X_b = np.vstack((X, np.ones(len(X)))).T
w_lls, b_lls = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y

print(f"LLS Solution: w = {w_lls:.4f}, b = {b_lls:.4f}")

def train_lms(x_data, y_data, lr, epochs):
    w = np.random.randn()
    b = np.random.randn()
    
    w_history = []
    b_history = []
    
    for _ in range(epochs):
        for i in range(len(x_data)):
            xi = x_data[i]
            yi = y_data[i]
            
            y_pred = w * xi + b
            error = yi - y_pred
            
            w += lr * error * xi
            b += lr * error * 1.0
            
        w_history.append(w)
        b_history.append(b)
        
    return w, b, w_history, b_history

lr = 0.01
epochs = 100
w_lms, b_lms, w_hist, b_hist = train_lms(X, Y, lr, epochs)

print(f"LMS Solution (lr={lr}): w = {w_lms:.4f}, b = {b_lms:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X, Y, color='red', label='Data Points')
x_line = np.linspace(0, 6, 100)
plt.plot(x_line, w_lls * x_line + b_lls, 'g-', linewidth=2, label=f'LLS (Exact)\n$y={w_lls:.2f}x+{b_lls:.2f}$')
plt.plot(x_line, w_lms * x_line + b_lms, 'b--', linewidth=2, label=f'LMS (Approx)\n$y={w_lms:.2f}x+{b_lms:.2f}$')
plt.title('Part (a) & (b): Linear Model Fitting')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(w_hist, label='Weight w')
plt.plot(b_hist, label='Bias b')
plt.axhline(y=w_lls, color='g', linestyle=':', label='LLS Optimal w')
plt.axhline(y=b_lls, color='orange', linestyle=':', label='LLS Optimal b')
plt.title('Part (b): Weight Convergence')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
lrs = [0.001, 0.01, 0.05]
for learning_rate in lrs:
    _, _, w_h, _ = train_lms(X, Y, learning_rate, 100)
    plt.plot(w_h, label=f'LR={learning_rate}')
plt.axhline(y=w_lls, color='k', linestyle='--', label='Optimal w')
plt.title('Part (d): Effect of Learning Rate on w')
plt.xlabel('Epochs')
plt.ylabel('Weight w')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()