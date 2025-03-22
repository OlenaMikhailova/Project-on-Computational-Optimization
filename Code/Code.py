# libraries
import numpy as np
import matplotlib.pyplot as plt

# 2D
def f(x):
    return x**2 + 4*x + 4

def df(x):
    return 2*x + 4

def gradient_descent_2d(learning_rate=0.1, num_iterations=100, start_point=-5):
    x = start_point
    x_values = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * df(x)  
        x_values.append(x)
    return x_values

x = np.linspace(-6, 2, 400)
y = f(x)

x_values = gradient_descent_2d(learning_rate=0.1, num_iterations=20, start_point=-5)


plt.plot(x, y, label="f(x) = x^2 + 4x + 4")
plt.scatter(x_values, [f(x) for x in x_values], color="red", label="Позиції під час спуску")
plt.plot(x_values, [f(x) for x in x_values], linestyle="--", color="red")
plt.title("Градієнтний спуск для f(x) = x^2 + 4x + 4")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

# 3D
def f_2d(x, y):
    return x**2 + y**2

def df_2d(x, y):
    return np.array([2*x, 2*y])

def gradient_descent_3d(learning_rate=0.1, num_iterations=100, start_point=np.array([2, 2])):
    point = start_point
    points = [point]
    for _ in range(num_iterations):
        gradient = df_2d(point[0], point[1])
        point = point - learning_rate * gradient 
        points.append(point)
    return np.array(points)

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f_2d(X, Y)

points_2d = gradient_descent_3d(learning_rate=0.1, num_iterations=20, start_point=np.array([2, 2]))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.plot(points_2d[:, 0], points_2d[:, 1], f_2d(points_2d[:, 0], points_2d[:, 1]), 
        color='red', marker='o', linestyle='-', markersize=5, label="Траєкторія спуску")

ax.set_title("Градієнтний спуск для f(x, y) = x^2 + y^2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

plt.legend()
plt.show()

