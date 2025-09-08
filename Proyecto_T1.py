#Credo por:
# - Brandon Ismael Rodriguez Rodriguez 23480289
# - Carlos Daniel Garcia Hernandez 24480704
# - Jorge Orlando Navarro Estrada 24480705
# - Victor Alan Pulido Guajardo 24480708
# - Jaime Isaias Villareal Gonzalez 24480709
from pulp import * #Libreria PuLP para programacion lineal
from itertools import combinations #Libreria para combinaciones
import matplotlib.pyplot as plt #Libreria Matplotlib para graficas
import numpy as np #Libreria Numpy para calculos numericos

#Definir el problema 
print("=== Funcion objetivo: Z = v1*x + v2*y ===")
v1 = float(input("Ingresa el valor de x: "))
v2 = float(input("Ingresa el valor de y: "))

#Sacar las restricciones
restricciones = []
num_res = int(input("¿Cuantas restricciones tienes? "))

for i in range(num_res):#Pide las restricciones
    print(f"Restriccion {i+1}: a*x + b*y <= c")
    a = float(input("  Coeficiente de x (a): "))
    b = float(input("  Coeficiente de y (b): "))
    c = float(input("  Disponible (c): "))
    restricciones.append((a, b, c))#Agrega las restricciones a la lista

#Calcular los puntos de interseccion
puntos = []
for r1, r2 in combinations(restricciones, 2): #Combinaciones de restricciones
    A = np.array([[r1[0], r1[1]], [r2[0], r2[1]]]) #Matriz de coeficientes
    B = np.array([r1[2], r2[2]]) #Vector de terminos independientes
    try:
        punto = np.linalg.solve(A, B) #Resolver el sistema de ecuaciones Ax = B
        #Verificar que cumpla todas las restricciones
        if all(r[0]*punto[0] + r[1]*punto[1] <= r[2] for r in restricciones):
            puntos.append(punto) #Agregar punto si cumple restricciones
    except np.linalg.LinAlgError:
        pass #Ignorar sistemas paralelos o dependientes

#Resolver con PuLP
prob = LpProblem("Maximizar_Z", LpMaximize)
x = LpVariable("x", lowBound=0)
y = LpVariable("y", lowBound=0)

#Funcion objetivo
prob += v1*x + v2*y
#Restricciones
for a, b, c in restricciones:
    prob += a*x + b*y <= c

prob.solve() #Resuelve el problema
x_opt = x.value() #Valor optimo de x
y_opt = y.value() #Valor optimo de y
z_opt = prob.objective.value() #Valor optimo de Z

print("\n=== Solución óptima ===")
print(f"x = {x_opt:.2f}")
print(f"y = {y_opt:.2f}")
print(f"Z máxima = {z_opt:.2f}")

#Conlusion
print("\nConclusión:")
print(f"Se debe producir {x_opt:.2f} unidades de x y {y_opt:.2f} unidades de y, logrando un valor máximo de Z = {z_opt:.2f}.")

#Graficar los resultados
#Calcular los limites de la grafica
x_max = 0
y_max = 0
for a, b, c in restricciones:
    if a != 0:
        x_max = max(x_max, c/a)
    if b != 0:
        y_max = max(y_max, c/b)
x_vals = np.linspace(0, x_max*1.2, 200)

plt.figure(figsize=(8, 8))

#Graficar cada restriccion
for a, b, c in (restricciones):
    if b != 0:
        y_vals = (c - a*x_vals) / b
        y_vals = np.clip(y_vals, 0, y_max*1.2)#Limitar valores negativos
        plt.plot(x_vals, y_vals, label=f"{a}x + {b}y <= {c}")#dibujar la linea
    else:
        plt.axvline(x=c/a, label=f"{a}x <= {c}")#dibujar linea vertical

    # Puntos de corte con los ejes
    pts = []
    if a != 0:
        x_cut = c / a
        pts.append((x_cut, 0))
    if b != 0:
        y_cut = c / b
        pts.append((0, y_cut))

    for px, py in pts:
        plt.plot(px, py, 'bs')  # puntos azules
        plt.text(px, py, f"({px:.1f},{py:.1f})", fontsize=8, color='blue')

# Puntos de intersección (excepto la solución óptima)
# Si no se repite la solucion optima en la grafica
for p in puntos:
    if not (np.isclose(p[0], x_opt) and np.isclose(p[1], y_opt)):
        plt.plot(p[0], p[1], "ko")  # puntos negros
        plt.text(p[0], p[1], f"({p[0]:.1f},{p[1]:.1f})", fontsize=8, color='black')

# Solución óptima con offset para texto
plt.plot(x_opt, y_opt, "ro", markersize=10, label="Solución óptima")
plt.text(x_opt + x_max*0.02, y_opt + y_max*0.02, f"({x_opt:.1f},{y_opt:.1f})", fontsize=10, color='red')

# Etiquetas y leyenda
plt.xlabel("x")
plt.ylabel("y")
plt.title("Grafica de restricciones y solución óptima")
plt.legend()
plt.grid(True)
plt.xlim(0, x_max*1.2)
plt.ylim(0, y_max*1.2)
plt.show()
