# IA-2024
# Arturo Navarro Velázquez

## Contenido
1. [Requerimientos ](#requerimientos)
2. [Explicación del código](#explicación-del-código)
3. [Resultados](#resultados)

# Proyecto 1 - Algoritmo Asterísco

El siguiente código es la construcción un programa empleando el método de *Búsqueda A* *.

Este algoritmo utiliza una combinación de funciones de costo y funciones heurísticas para encontrar la solución óptima.

## Requerimientos

- Tener la vesión de python 3.10 o superior
- Instalar la libreria de 'pygame' con el siguiente comando.

``` python
pip install pygame
```
- Instalar la libreria de 'tkinter' con el siguiente comando.
``` python
pip install tk
```
- Instalar la libreria de 'PriorityQueue' con el siguiente comando.
``` python
pip install PriorityQueue
```


Pygame es una librería popular para la creación de videojuegos y simulaciones en Python. Proporciona herramientas para el manejo de gráficos, sonido, eventos de teclado y ratón, y otros elementos necesarios para construir interfaces visuales interactivas.

## Explicación del código

Primeramente se definen los siguiente colores y el ancho de la ventana.

``` python

import pygame
from queue import PriorityQueue
import tkinter as tk
from threading import Thread
import string

# Configuraciones iniciales
ANCHO_VENTANA = 800

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

pygame.font.init()
FUENTE = pygame.font.SysFont("comicsans", 50)


```

Seguido creamos una clase clase Nodo para controlar las operaciones:

- Contruir la ventana
- Construir el camino
- Dibuja el nodo en la ventana con su color actual
- Restablecer
- Crear las paredes de color negro
- Colorear el inicio
- Colorear de morado el final
- Actualizar_vecinos: Determina qué nodos adyacentes (vecinos) son accesibles. Soporta movimientos diagonales además de los clásicos "arriba", "abajo", "izquierda" y "derecha"

``` python
class Nodo:
    def __init__(self, fila, col, ancho, total_filas, etiqueta):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        self.padre = None
        self.etiqueta = etiqueta

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # Movimiento abajo
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        # Movimiento arriba
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        # Movimiento derecha
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        # Movimiento izquierda
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])
        # Movimiento diagonal abajo-derecha
        if self.fila < self.total_filas - 1 and self.col < self.total_filas - 1 and not grid[self.fila + 1][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col + 1])
        # Movimiento diagonal abajo-izquierda
        if self.fila < self.total_filas - 1 and self.col > 0 and not grid[self.fila + 1][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col - 1])
        # Movimiento diagonal arriba-derecha
        if self.fila > 0 and self.col < self.total_filas - 1 and not grid[self.fila - 1][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col + 1])
        # Movimiento diagonal arriba-izquierda
        if self.fila > 0 and self.col > 0 and not grid[self.fila - 1][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col - 1])

    def __lt__(self, other):
        return self.f < other.f

```

La clase ListaVentana muestra **la lista abierta** y **la lista cerradas** en tiempo real. <br>
La lista abierta contiene los nodos candidatos a ser explorados y la lista cerrada contiene los nodos ya explorados.

``` python

class ListaVentana:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Listas Abierta y Cerrada")
        self.root.geometry("400x600")

        self.label_abierta = tk.Label(self.root, text="Lista Abierta:", font=("Arial", 14, "bold"))
        self.label_abierta.pack()
        self.text_abierta = tk.Text(self.root, height=15, width=50)
        self.text_abierta.pack()

        self.label_cerrada = tk.Label(self.root, text="Lista Cerrada:", font=("Arial", 14, "bold"))
        self.label_cerrada.pack()
        self.text_cerrada = tk.Text(self.root, height=15, width=50)
        self.text_cerrada.pack()

    def actualizar_abierta(self, abierta):
        self.text_abierta.delete(1.0, tk.END)
        for nodo in abierta:
            if nodo.padre:
                self.text_abierta.insert(tk.END, f"{nodo.padre.etiqueta} -> {nodo.etiqueta}\n")
            else:
                self.text_abierta.insert(tk.END, f"{nodo.etiqueta}\n")
        self.text_abierta.update_idletasks()

    def actualizar_cerrada(self, cerrada):
        self.text_cerrada.delete(1.0, tk.END)
        for nodo in cerrada:
            if nodo.padre:
                self.text_cerrada.insert(tk.END, f"{nodo.padre.etiqueta} -> {nodo.etiqueta}\n")
            else:
                self.text_cerrada.insert(tk.END, f"{nodo.etiqueta}\n")
        self.text_cerrada.update_idletasks()

    def iniciar(self):
        self.root.mainloop()


```

Creamos una función la dibujar los asteríscos

``` python

def a_star(dibujar, grid, inicio, fin, lista_ventana):
    cont = 0
    open_set = PriorityQueue()
    open_set.put((0, cont, inicio))
    came_from = {}

    inicio.g = 0
    inicio.f = heuristica(inicio, fin)

    open_set_hash = {inicio}
    closed_set = set()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        actual = open_set.get()[2]
        open_set_hash.remove(actual)
        closed_set.add(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            return True

        for vecino in actual.vecinos:
            diagonal = abs(vecino.fila - actual.fila) == 1 and abs(vecino.col - actual.col) == 1
            temp_g_score = actual.g + (1.4 if diagonal else 1)

            if temp_g_score < vecino.g:
                vecino.padre = actual
                came_from[vecino] = actual
                vecino.g = temp_g_score
                vecino.h = heuristica(vecino, fin)
                vecino.f = vecino.g + vecino.h
                if vecino not in open_set_hash and vecino not in closed_set:
                    cont += 1
                    open_set.put((vecino.f, cont, vecino))
                    open_set_hash.add(vecino)
                    vecino.color = ROJO

        dibujar()
        actual.color = AZUL
        pygame.time.delay(30)

        lista_ventana.actualizar_abierta(open_set_hash)
        lista_ventana.actualizar_cerrada(closed_set)

    return False



```

En esta función realizamos la heurística para para hallar los vecinos más cernanos en base a la fórmula:

F = G + H

**F:** El costo total.<br>
**G:** Costo de a los vencinos.<br>
**H:** Costo inicial al punto final sin muros.<br>

``` python

for vecino in actual.vecinos:
            diagonal = abs(vecino.fila - actual.fila) == 1 and abs(vecino.col - actual.col) == 1
            temp_g_score = actual.g + (1.4 if diagonal else 1)

            if temp_g_score < vecino.g:
                vecino.padre = actual
                came_from[vecino] = actual
                vecino.g = temp_g_score
                vecino.h = heuristica(vecino, fin)
                vecino.f = vecino.g + vecino.h
                if vecino not in open_set_hash and vecino not in closed_set:
                    cont += 1
                    open_set.put((vecino.f, cont, vecino))
                    open_set_hash.add(vecino)
                    vecino.color = ROJO
```

Finalmente se crea la función main para ejecutar el programa.

``` pythton

def ejecutar_pygame(lista_ventana):
    ventana = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
    filas = 20
    grid = crear_grid(filas, ANCHO_VENTANA)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, filas, ANCHO_VENTANA)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, filas, ANCHO_VENTANA)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, filas, ANCHO_VENTANA)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    a_star(lambda: dibujar(ventana, grid, filas, ANCHO_VENTANA), grid, inicio, fin, lista_ventana)

                if event.key == pygame.K_r:
                    grid = crear_grid(filas, ANCHO_VENTANA)
                    inicio = None
                    fin = None

                if event.key == pygame.K_q:
                    corriendo = False

    pygame.quit()


```

## Resultados

Las siguientes imágenes son los resultados del programa


![Imagen del proceso del A*](./evidencias/evidenciasasterisco/Captura%20de%20pantalla%202024-12-18%20191652.png)
![Imagen del proceso del A*](./evidencias/evidenciasasterisco/Captura%20de%20pantalla%202024-12-18%20194435.png)
![Imagen del proceso del A*](./evidencias/evidenciasasterisco/Captura%20de%20pantalla%202024-12-18%20194818.png)
