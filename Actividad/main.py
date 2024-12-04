import numpy as np
import matplotlib.pyplot as plt


data = np.array([
    [0.5, 0.8, 1],
    [0.6, 0.9, 1],
    [0.7, 0.6, 0],
    [0.4, 0.5, 0],
    [0.3, 0.9, 1],
    [0.8, 0.4, 0]
])

X = data[:, :2]  
y = data[:, 2]  


w = np.random.rand(2)  
b = np.random.rand()   
learning_rate = 0.1    
epochs = 100          


def step_function(z):
    return 1 if z >= 0 else 0


for epoch in range(epochs):
    for i in range(len(X)):
        # Calcular la salida
        z = np.dot(X[i], w) + b
        y_pred = step_function(z)
        # Actualizar pesos y sesgo si hay error
        error = y[i] - y_pred
        w += learning_rate * error * X[i]
        b += learning_rate * error


print("Pesos finales:", w)
print("Sesgo final:", b)


x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.array([step_function(np.dot([x, y], w) + b) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="coolwarm")
plt.title("Frontera de Decisión del Perceptrón")
plt.xlabel("Precio Relativo")
plt.ylabel("Calidad Percibida")
plt.show()


print("¿Datos linealmente separables? Sí, debido a que el perceptrón converge para esta tarea.")
