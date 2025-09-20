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
# Se usa session_state para guardar las restricciones dinámicamente
if "restricciones" not in st.session_state:
    st.session_state.restricciones = []

# Formulario para agregar nuevas restricciones
with st.form("agregar_restriccion_form"):
    st.write("Agregar una nueva restricción: a*x + b*y <= c")
    a = st.number_input("Coeficiente de x (a)", value=1.0)
    b = st.number_input("Coeficiente de y (b)", value=1.0)
    c = st.number_input("Disponible (c)", value=10.0)
    add_btn = st.form_submit_button("Agregar restricción") #Botón para agregar restricción
    if add_btn:
        st.session_state.restricciones.append((a, b, c)) #Agrega la restricción a la lista

# Mostrar todas las restricciones actuales
st.subheader("Restricciones actuales:")
for idx, (a, b, c) in enumerate(st.session_state.restricciones):
    st.write(f"{idx+1}: {a}*x + {b}*y <= {c}")

# Botón para procesar y resolver el problema
if st.button("Resolver problema"):
    restricciones = st.session_state.restricciones.copy()
    if len(restricciones) < 1:
        st.warning("Agrega al menos una restricción antes de resolver.") #Validación
    else:
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
        x_max = max([c/a if a != 0 else 0 for a,b,c in restricciones])*1.2
        y_max = max([c/b if b != 0 else 0 for a,b,c in restricciones])*1.2
        x_vals = np.linspace(0, x_max, 400)
        fig, ax = plt.subplots(figsize=(8,8))
        # Graficar restricciones
        for a, b, c in restricciones:
            if b != 0:
                y_vals = np.clip((c - a*x_vals)/b, 0, y_max)
                ax.plot(x_vals, y_vals, label=f"{a}x + {b}y <= {c}")
            else:
                ax.axvline(c/a, label=f"{a}x <= {c}")

        # Puntos de intersección (excepto solución óptima)
        for p in puntos:
            if not (np.isclose(p[0], x_opt) and np.isclose(p[1], y_opt)):
                ax.plot(p[0], p[1], "ko")

        # Solución óptima
        ax.plot(x_opt, y_opt, "ro", markersize=10, label="Solución óptima")

        # Etiquetas y leyenda
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Región factible con solución óptima")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

        st.pyplot(fig) # Mostrar gráfico en Streamlit

        st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <hr>
            <p><b>Creado por:</b></p>
            <p>- Brandon Ismael Rodriguez Rodriguez 23480289</p>
            <p>- Carlos Daniel Garcia Hernandez 24480704</p>
            <p>- Jorge Orlando Navarro Estrada 24480705</p>
            <p>- Victor Alan Pulido Guajardo 24480708</p> 
            <p>- Jaime Isaias Villareal Gonzalez 24480709</p>
            <p>© 2025 Todos los derechos reservados.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
