import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog as fd #no va sin esta libreria
import tensorflow as tf

#Lista con los nombres de la clase CIFAR-10 (traducidos al español)
nombres_clase = ["Avión", "Automóvil", "Pájaro", "Gato", "Ciervo", "Perro", "Rana", "Caballo", "Barco", "Camión"]

#Carga modelo ya entrenado por un archivo en una ruta determinada
def carga_modelo():
    modelo = tf.keras.models.load_model("C:\\Users\\Kirian\\Downloads\\modelo_completo.h5")
    return modelo

#Recibe una imagen y el modelo ya cargado
def importacion_prediccion(datos_imagen, modelo):
    #Ajuste a 32x32
    tamaño = (32, 32)
    imagen = ImageOps.fit(datos_imagen, tamaño, Image.LANCZOS)
    #Se convierte en array
    img = np.asarray(imagen)
    #Reajuste para que sea compatible con el modelo
    img_reshape = img[np.newaxis, ...]
    #Prediccion
    prediccion = modelo.predict(img_reshape)
    return prediccion

#Permite seleccionar el archivo mediante el comando 'askopenfilename' 
def explorar_archivo():
    global ruta_imagen
    ruta_imagen = fd.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png")])
    if ruta_imagen:
        imagen = Image.open(ruta_imagen)
        imagen = imagen.resize((300, 300))  # Ajustar el tamaño de la imagen para que se vea correctamente
        imagen_foto = ImageTk.PhotoImage(imagen) # Hace compatible la imagen con tkinter
        etiqueta_imagen.configure(image=imagen_foto) #Incorpora la imagen
        etiqueta_imagen.image = imagen_foto  # Mantener una referencia a la imagen para evitar que se elimine de la memoria
        habilitar_boton_prediccion() #Activa el boton

def habilitar_boton_prediccion(): 
    boton_prediccion.configure(state="normal", command=predecir_imagen)

#Al presionar el anterior boton, se activa esta funcion
def predecir_imagen():
    #Abre imagen ya seleccionada
    imagen = Image.open(ruta_imagen)
    #Predicciones incorporadas
    predicciones = importacion_prediccion(imagen, modelo)
    #indice_clase -> Indica de que clase es una imagen segun prediccion. ex: indice_clase[0] = avión 
        #argmax para mayor probabilidad
    indice_clase = np.argmax(predicciones)
    clase_predicha = nombres_clase[indice_clase]
    #Actualizacion texto
    etiqueta_resultado.configure(text="La imagen se parece a: " + clase_predicha)

# Crea la app con tkinter
app = tk.Tk()
app.title("Clasificación de imágenes con el conjunto de datos CIFAR-10")

# Carga del modelo
modelo = carga_modelo()

# Crear los elementos de la interfaz gráfica
etiqueta_titulo = tk.Label(app, text="Clasificación de imágenes con el conjunto de datos CIFAR-10", font=("Helvetica", 18, "bold"))

etiqueta_titulo.pack(pady=10)

etiqueta_encabezado = tk.Label(app, text="Cargue imágenes de los siguientes elementos:", font=("Helvetica", 14))
etiqueta_encabezado.pack()

etiqueta_clase = tk.Label(app, text=nombres_clase, font=("Helvetica", 12))
etiqueta_clase.pack()

boton_explorar = tk.Button(app, text="Cargar imagen", command=explorar_archivo, font=("Helvetica", 10))
boton_explorar.pack(pady=10)

etiqueta_imagen = tk.Label(app)
etiqueta_imagen.pack()

boton_prediccion = tk.Button(app, text="Predecir", state="disabled", font=("Helvetica", 10))
boton_prediccion.pack(pady=10)

etiqueta_resultado = tk.Label(app, font=("Helvetica", 10))
etiqueta_resultado.pack(pady=10)

# Ejecutar aplicacion
app.mainloop()