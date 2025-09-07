#Importar la libreria PuLP
from pulp import *
#Definir el problema
prob = LpProblem("Maximimizacion_de_beneficios", LpMaximize)
#Definir las variables de decision
x = LpVariable('Producto_A', lowBound=0)
y = LpVariable('Producto_B', lowBound=0)
#Establecer la funcion objetivo
prob += 3 * x + 2 * y, "Beneficio_total"
#Agregar las restricciones
prob += 2 * x + y <= 100, "Restriccion_Recurso_1"
prob += x + 2 * y <= 80, "Restriccion_Recurso_2"
#Resolver el problema
prob.solve()
#Imprimir los resultados
print("Estado:", LpStatus[prob.status])
print(f"Produccion de Producto A:", x.varValue)
print(f"Produccion de Producto B:", y.varValue)
print(f"Beneficio total: ${value(prob.objective)}")