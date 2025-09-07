import streamlit as st #Liberería para crear aplicaciones web interactivas
import numpy as np #Librería Numpy para cálculos numéricos
import matplotlib.pyplot as plt #Librería Matplotlib para gráficos
from itertools import combinations #Librería para combinaciones
from pulp import * #Librería PuLP para programación lineal

st.title("Proyecto T1: Programación Lineal Interactiva") #Título de la aplicación

# Función objetivo
st.header("Función Objetivo: Z = v1*x + v2*y") 
v1 = st.number_input("Coeficiente de x (v1)", value=1.0)
v2 = st.number_input("Coeficiente de y (v2)", value=1.0)

# Restricciones
st.header("Restricciones")
num_res = st.number_input("¿Cuántas restricciones tienes?", min_value=1, value=2, step=1)#Formulario para restricciones
restricciones = []

with st.form("restricciones_form"):
    for i in range(int(num_res)):#Pide las restricciones
        st.write(f"Restricción {i+1}: a*x + b*y <= c")
        a = st.number_input(f"Coeficiente de x (a) restricción {i+1}", value=1.0)
        b = st.number_input(f"Coeficiente de y (b) restricción {i+1}", value=1.0)
        c = st.number_input(f"Disponible (c) {i+1}", value=10.0)
        restricciones.append((a, b, c))
    
    submitted = st.form_submit_button("Resolver") #Botorn para enviar formulario

# Procesar datos al enviar formulario
if submitted:
    # Calcular puntos de intersección
    puntos = []
    for r1, r2 in combinations(restricciones, 2): # Combinaciones de restricciones
        A = np.array([[r1[0], r1[1]], [r2[0], r2[1]]]) # Matriz de coeficientes
        B = np.array([r1[2], r2[2]]) # Vector de términos independientes
        try:
            punto = np.linalg.solve(A, B)  # Resolver el sistema de ecuaciones Ax = B
            # Verificar que cumpla todas las restricciones
            if all(r[0]*punto[0] + r[1]*punto[1] <= r[2] for r in restricciones):
                puntos.append(punto) # Agregar punto si cumple restricciones
        except np.linalg.LinAlgError: #Ignora sistemas paralelos o dependientes
            pass

    # Resolver con PuLP
    prob = LpProblem("Maximizar_Z", LpMaximize)
    x = LpVariable("x", lowBound=0)
    y = LpVariable("y", lowBound=0)
    # Función objetivo
    prob += v1*x + v2*y
    # Restricciones
    for a, b, c in restricciones:
        prob += a*x + b*y <= c

    prob.solve() # Resuelve el problema
    x_opt = x.value() #Valor óptimo de x
    y_opt = y.value() #Valor óptimo de y
    z_opt = prob.objective.value() #Valor óptimo de Z

    st.subheader("Conclusión:") #Muestra la conclusión
    st.write(f"Se debe producir {x_opt:.2f} unidades de x y {y_opt:.2f} unidades de y, logrando un valor máximo de Z = {z_opt:.2f}.")

    # Graficar los resultados
    # Determinar límites para la gráfica
    x_max = 0
    y_max = 0
    for a, b, c in restricciones:
        if a != 0:
            x_max = max(x_max, c / a)
        if b != 0:
            y_max = max(y_max, c / b)

    x_vals = np.linspace(0, x_max*1.2, 400)
    fig, ax = plt.subplots(figsize=(8,8))
    # Graficar restricciones
    for a, b, c in restricciones:
        if b != 0:
            y_vals = (c - a*x_vals)/b
            y_vals = np.clip(y_vals, 0, y_max*1.2)
            ax.plot(x_vals, y_vals, label=f"{a}x + {b}y <= {c}")
        else:
            ax.axvline(c/a, label=f"{a}x <= {c}")

        # Cortes con ejes
        pts = []
        if a != 0:
            x_cut = c / a
            pts.append((x_cut, 0))
        if b != 0:
            y_cut = c / b
            pts.append((0, y_cut))
        for px, py in pts:
            ax.plot(px, py, 'bs')
            ax.text(px, py, f"({px:.1f},{py:.1f})", fontsize=8, color='blue')

    # Puntos de intersección (excepto solución óptima)
    for p in puntos:
        if not (np.isclose(p[0], x_opt) and np.isclose(p[1], y_opt)):
            ax.plot(p[0], p[1], "ko")
            ax.text(p[0], p[1], f"({p[0]:.1f},{p[1]:.1f})", fontsize=8, color='black')

    # Solución óptima
    ax.plot(x_opt, y_opt, "ro", markersize=10, label="Solución óptima")
    ax.text(x_opt + x_max*0.02, y_opt + y_max*0.02, f"({x_opt:.1f},{y_opt:.1f})", fontsize=10, color='red')

    # Etiquetas y leyenda
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Región factible con cortes, intersecciones y solución óptima")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, x_max*1.2)
    ax.set_ylim(0, y_max*1.2)

    st.pyplot(fig) # Mostrar gráfico en Streamlit