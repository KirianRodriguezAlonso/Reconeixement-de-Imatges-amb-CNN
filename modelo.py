import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def crear_modelo():

    #Creacion de la secuencia
    modelo = Sequential()
    #Capa convolucional de 32 filtros 3x3. padding = 'same' para que el output tenga el mismo tamaño que input. ReLU para 0 o 1
    modelo.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3), activation = 'relu'))
    modelo.add(Conv2D(32,(3,3), activation = 'relu'))
    #Agrega Capa Maxpooling
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    #Evita overfitting
    modelo.add(Dropout(0.25))

    modelo.add(Conv2D(64,(3,3),padding='same', activation = 'relu'))
    modelo.add(Conv2D(64,(3,3), activation = 'relu'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Dropout(0.25))

    #Capa Flatten o de aplanamiento para pasar de mapas 2D a vector 1D
    modelo.add(Flatten())
    #Fully connected para ayudar a aprender las caracteristicas y relaciones mas complejas
    modelo.add(Dense(512, activation = 'relu'))
    modelo.add(Dropout(0.25))
    #Softmax para sacar la distribucion de probabilidad, tamaño 10 -> Cifar-10 tiene 10 clases
    modelo.add(Dense(10, activation = 'softmax'))
    
    return modelo

def entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    #Compilacion del modelo anterior, 'categorical_crossentropy' -> clasificacion de multiples clases
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Entrenamiento especificando los hiperparametros necesarios, este training se guarda en 'historial'
    historial = modelo.fit(x_entrenamiento, y_entrenamiento, batch_size=200, epochs=10, validation_data=(x_prueba, y_prueba))

    # Extraen las métricas de entrenamiento y prueba del historial de entrenamiento
    perdida_entrenamiento = historial.history['loss']
    perdida_prueba = historial.history['val_loss']
    precision_entrenamiento = historial.history['accuracy']
    precision_prueba = historial.history['val_accuracy']

    # Representar gráficamente las curvas de error de entrenamiento y prueba
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(perdida_entrenamiento, label='Entrenamiento')
    plt.plot(perdida_prueba, label='Prueba')
    plt.title('Curva de error')
    plt.xlabel('Épocas')
    plt.ylabel('Error')
    plt.legend()

    # Representar gráficamente la curva de precisión de prueba
    plt.subplot(1, 2, 2)
    plt.plot(precision_prueba)
    plt.title('Curva de precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')

    plt.tight_layout()
    plt.show()
    
    # Definicion de las Id de las clases
    cifar10_clases = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

    # Obtencion de las predicciones del modelo utilizando los datos de prueba
    y_prediccion = modelo.predict(x_prueba)
    # Argmax -> obtiene los índices con mas % de acierto
    y_prediccion = np.argmax(y_prediccion, axis=1)
    # Se compara las predicciones con los valores reales
    precision_clases = accuracy_score(np.argmax(y_prueba, axis=1), y_prediccion)
    # Resultados de cada clase
    reporte_clases = classification_report(np.argmax(y_prueba, axis=1), y_prediccion, output_dict=True)

    # Mostrar la precisión de cada clase
    for class_id, class_metrics in reporte_clases.items():
      if class_id.isdigit(): # Necesario ya que 'accuracy' se encuentra dentro de reporte_clases.items()
        class_name = cifar10_clases[int(class_id)]
        accuracy = class_metrics['precision']
        print(f'Precisión de la clase {class_name}: {accuracy * 100:.2f}%')
    


def main():
  
    # Cargar el dataset CIFAR-10
    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cifar10.load_data()

    # Normalizar las imágenes (0-1) y 
    x_entrenamiento = x_entrenamiento.astype('float32') / 255
    x_prueba = x_prueba.astype('float32') / 255

    # Convertir las etiquetas a one-hot encoding 
        # One-hot -> Convierte las etiquetas en vectores binarios de tamaño 10 donde  el índice de la clase es 1 y el resto  0
    y_entrenamiento = to_categorical(y_entrenamiento)
    y_prueba = to_categorical(y_prueba)

    # Crear el modelo de la CNN
    modelo = crear_modelo()

    # Entrenar el modelo y obtener métricas
    entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)
    
    # Guardar modelo
    modelo.save("./modelo_completo.h5")
    print("Modelo guardado correctamente.")

if __name__ == '__main__': 
    main()
