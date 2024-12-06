import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tkinter import Tk, Button, filedialog, Label, Entry
from PIL import Image, ImageTk
import urllib.request
from io import BytesIO

# modelo previamente entrenado
model = load_model('modelo_cars.h5')

# Clases
clases = [
    'Convertible', 'Coupe', 'Electric', 'Sedan',
    'Sport', 'SUV', 'Truck', 'Van', 'Wagon'
]

# Variable global para detener la cámara
capturando = False

# preprocesar la imagen
def preprocesar_imagen(frame, tamaño=(150, 150)):
    img_resized = cv2.resize(frame, tamaño)
    img_normalizada = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalizada, axis=0)
    return img_expanded

# procesar y mostrar la predicción de una imagen
def procesar_y_mostrar(frame):
    img_preprocesada = preprocesar_imagen(frame)
    pred = model.predict(img_preprocesada)
    clase_predicha = np.argmax(pred, axis=1)[0]
    clase_nombre = clases[clase_predicha]

    # resultado
    texto = f"Tipo de Auto: {clase_nombre}"
    print(texto)

    # Convertir frame a formato RGB para Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((400, 300))  # Tamaño reducido
    img_tk = ImageTk.PhotoImage(img)

    # Mostrar imagen en la interfaz
    label_img.configure(image=img_tk)
    label_img.image = img_tk

    # Actualizar el resultado en el label
    label_resultado.config(text=texto)

# cargar  imagen desde el ordenador
def cargar_imagen():
    file_path = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        procesar_y_mostrar(img)

# cargar imagen desde una URL
def cargar_imagen_url():
    url = entry_url.get()  # Obtener la URL de la imagen
    if url:
        try:
            # imagen desde la URL
            resp = urllib.request.urlopen(url)
            img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            procesar_y_mostrar(img)
        except Exception as e:
            print(f"Error al cargar la imagen desde la URL: {e}")
            label_resultado.config(text="Error al cargar la imagen desde la URL.")

# encender la cámara y realizar predicciones en tiempo real
def encender_camara():
    global capturando
    capturando = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    def procesar_frame():
        if not capturando:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            cap.release()
            return

        # predicción
        img_preprocesada = preprocesar_imagen(frame)
        pred = model.predict(img_preprocesada)
        clase_predicha = np.argmax(pred, axis=1)[0]
        clase_nombre = clases[clase_predicha]

        # predicción en el frame
        texto = f"Tipo de Auto: {clase_nombre}"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convertir frame a formato RGB para Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((800, 400))  # Tamaño reducido
        img_tk = ImageTk.PhotoImage(img)

        # Mostrar imagen en la interfaz
        label_img.configure(image=img_tk)
        label_img.image = img_tk

        # Actualizar el resultado en el label
        label_resultado.config(text=texto)

        # Continuar capturando frames
        label_img.after(10, procesar_frame)

    procesar_frame()

# detener la cámara
def detener_camara():
    global capturando
    capturando = False
    label_img.config(image='')  # Limpia la imagen en el label
    label_resultado.config(text="Resultado")  # Resetea el resultado

# Crear ventana Tkinter
ventana = Tk()
ventana.title("Clasificación de Automóviles")

# Ajustar tamaño de ventana
ventana.geometry("600x600")
ventana.config(bg="#f5f5f5")

# Título de la ventana
label_titulo = Label(ventana, text="Clasificación de Automóviles", font=("Helvetica", 18, "bold"), bg="#f5f5f5", fg="#333")
label_titulo.pack(pady=10)

# Frame para botones
frame_botones = Label(ventana, bg="#f5f5f5")
frame_botones.pack(pady=10)

# Botón  cargar imagen
btn_cargar = Button(frame_botones, text="Cargar Imagen", command=cargar_imagen, width=15, height=1, font=("Helvetica", 10), bg="#4CAF50", fg="white", relief="raised")
btn_cargar.grid(row=0, column=0, padx=5, pady=5)

# Botón  encender la cámara
btn_camara = Button(frame_botones, text="Encender Cámara", command=encender_camara, width=15, height=1, font=("Helvetica", 10), bg="#2196F3", fg="white", relief="raised")
btn_camara.grid(row=0, column=1, padx=5, pady=5)

# Botón  detener la cámara
btn_detener = Button(frame_botones, text="Detener Cámara", command=detener_camara, width=15, height=1, font=("Helvetica", 10), bg="#f44336", fg="white", relief="raised")
btn_detener.grid(row=0, column=2, padx=5, pady=5)

# Campo para ingresar URL de la imagen
label_url = Label(ventana, text="Ingrese la URL de la imagen:", font=("Helvetica", 10), bg="#f5f5f5")
label_url.pack(pady=5)

entry_url = Entry(ventana, width=50, font=("Helvetica", 10))
entry_url.pack(pady=5)

# Botón para cargar imagen desde la URL
btn_url = Button(ventana, text="Cargar Imagen desde URL", command=cargar_imagen_url, width=20, height=1, font=("Helvetica", 10), bg="#ff9800", fg="white", relief="raised")
btn_url.pack(pady=10)

# Label  mostrar la imagen o video
label_img = Label(ventana, width=400, height=300, bg="#ddd", relief="sunken")
label_img.pack(pady=10)

# Label  mostrar  resultado
label_resultado = Label(ventana, text="Resultado", font=("Helvetica", 14), bg="#f5f5f5", fg="#333")
label_resultado.pack(pady=10)

# Iniciar ventana
ventana.mainloop()
