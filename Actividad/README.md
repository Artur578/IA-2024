# Clasificación de Productos según su Aceptación por Clientes

## Descripción del Problema
Una empresa desea predecir si un producto será aceptado o rechazado por los clientes basándose en las siguientes características:

- **Precio Relativo**: Precio del producto en comparación con productos similares (normalizado entre 0 y 1).
- **Calidad Percibida**: Opinión de los clientes sobre la calidad del producto en encuestas (normalizada entre 0 y 1).

### Criterios de Clasificación
Un producto se considera **aceptado** si:
- El precio relativo es menor o igual a 0.6.
- La calidad percibida es mayor o igual a 0.7.

En caso contrario, será **rechazado**.

---

## Modelo Implementado

### Entradas del Perceptrón
- \( x_1 \): Precio Relativo (0-1).
- \( x_2 \): Calidad Percibida (0-1).

### Salida del Perceptrón
- **1**: Aceptado.
- **0**: Rechazado.

---

## Datos de Entrenamiento

| Precio Relativo | Calidad Percibida | Resultado |
|------------------|-------------------|-----------|
| 0.5              | 0.8               | 1         |
| 0.6              | 0.9               | 1         |
| 0.7              | 0.6               | 0         |
| 0.4              | 0.5               | 0         |
| 0.3              | 0.9               | 1         |
| 0.8              | 0.4               | 0         |

---

## Implementación

El modelo es un perceptrón simple, implementado en Python. Los pasos incluyen:

1. **Inicialización de pesos y sesgo**: Los valores se inicializan aleatoriamente.
2. **Función de Activación**: Se usa una función escalón para decidir si el producto es aceptado o rechazado.
3. **Entrenamiento**: Los pesos y el sesgo se ajustan iterativamente con aprendizaje supervisado.
4. **Frontera de Decisión**: Se grafica la separación entre los productos aceptados y rechazados.

---

### Código

```python
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento
data = np.array([
    [0.5, 0.8, 1],
    [0.6, 0.9, 1],
    [0.7, 0.6, 0],
    [0.4, 0.5, 0],
    [0.3, 0.9, 1],
    [0.8, 0.4, 0]
])

X = data[:, :2]  # Características (precio relativo, calidad percibida)
y = data[:, 2]   # Etiquetas (1: Aceptado, 0: Rechazado)

# Parámetros del perceptrón
w = np.random.rand(2)  # Pesos iniciales
b = np.random.rand()   # Sesgo inicial
learning_rate = 0.1    # Tasa de aprendizaje
epochs = 100           # Número de iteraciones

# Función de activación
def step_function(z):
    return 1 if z >= 0 else 0

# Entrenamiento del perceptrón
for epoch in range(epochs):
    for i in range(len(X)):
        # Calcular la salida
        z = np.dot(X[i], w) + b
        y_pred = step_function(z)
        # Actualizar pesos y sesgo si hay error
        error = y[i] - y_pred
        w += learning_rate * error * X[i]
        b += learning_rate * error

# Resultados finales
print("Pesos finales:", w)
print("Sesgo final:", b)

# Graficar la frontera de decisión
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


Resultados
¿Son los datos linealmente separables?
Sí, los datos son linealmente separables, ya que el perceptrón converge y define una frontera de decisión clara entre los productos aceptados y rechazados.

¿Qué ajustes podrían hacer al modelo para mejorar la predicción?
Ajustar la tasa de aprendizaje para optimizar la convergencia.
Si los datos no fueran linealmente separables, considerar un modelo más avanzado como:
Redes Neuronales Multicapa (MLP).
Kernel Trick en un modelo SVM.
Aumentar el conjunto de datos para mejorar la generalización.
Describir cada una de las partes del modelo implementado
Inicialización: Se definen pesos (
𝑤
w) y sesgo (
𝑏
b) aleatoriamente.
Función de Activación: Decide si el producto es aceptado (
1
1) o rechazado (
0
0).
Entrenamiento:
Calcula la salida.
Ajusta pesos y sesgo si hay error en la predicción.
Frontera de Decisión: Grafica la línea que separa las predicciones aceptadas de las rechazadas.

### Gráfica de la Frontera de Decisión

La siguiente gráfica muestra la frontera de decisión aprendida por el perceptrón. Los puntos rojos corresponden a productos aceptados (clase 1) y los puntos azules a productos rechazados (clase 0). La región en **rojo** indica que el perceptrón clasifica como aceptado, mientras que la región en **azul** clasifica como rechazado.

![Frontera de Decisión](images/frontera_desicion.png)
