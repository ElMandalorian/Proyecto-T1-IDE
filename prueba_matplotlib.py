import matplotlib.pyplot as plt
#Crear la figura y los ejes
fig, ax = plt.subplots()
#Dibuja los puntos
ax.scatter(x = [1, 2, 3], y = [3, 2, 1])
ax.plot([1, 2, 3], [3, 2, 1], label='Linea de puntos')
#Guardar la grafica en formato png
plt.savefig("grafica.png")
#Mostrar la grafica
plt.show()