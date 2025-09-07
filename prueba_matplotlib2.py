import matplotlib.pyplot as plt
#Para varias figuras en una misma ventana
#fig, axs = plt.subplots(1, 2)
#axs[0].scatter([1, 2, 3],[3, 2, 1])#dibuja los puntos
#axs[0].set_title('Grafica Izquierda')

#axs[1].plot([1, 2, 3],[3, 2, 1])#Dibujar la linea
#axs[1].set_title('Grafica Derecha')
#plt.show()

#Mostrar varias figuras en ventanas separadas
plt.figure()
plt.scatter([1, 2, 3],[3, 2, 1], label='Punto 1')
plt.title('Grafica 1')
plt.legend()
plt.show()

plt.figure()
plt.plot([1, 2, 3],[3, 2, 1], label='Linea 2', color='red')
plt.title('Grafica 2')
plt.legend()
plt.show()