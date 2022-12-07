from tkinter import *
from tkinter import messagebox, ttk
import tkinter as tk
import os
import shutil
import cv2
import tkinter.messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
import imutils
import numpy as np


import math
import glob

import matplotlib.pyplot as plt


image_face = None

def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".bmp"),
        ("image", ".jpg")])
    if len(path_image) > 0:
        global image
        #path
        global image_face
        image_face = path_image
        print(image_face)
        # Leer la imagen de entrada y la redimensionamos
        image = cv2.imread(path_image)
        image= imutils.resize(image, height=380)
        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=180)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)
        lblInputImage.configure(image=img)
        lblInputImage.image = img

def guardar_huella():
    imagen = Image.open(image_face)
    imagen = imagen.resize((550,550))
    imagen.save("huella.bmp")
    tkinter.messagebox.showinfo(title="Alerta",message="Imagen Procesada ")
#se guarda todos los resultados de las comparaciones en porcentaje
resultados = []
#donde se guardan los nombres de las personas
persona = []
titulos = []
def buscar_coincidencias():
    #persona.clear()
    global titulos
    original = cv2.imread("huella.bmp")
    # lectura de todas imagenes
    # Sift and Flann
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #recorriendo por las rutas
    # Load all the images
    all_images_to_compare = []
    titles = []
    for f in glob.iglob("img/*"):
        image = cv2.imread(f)
        titles.append(f)
        titulos.append(f)
        all_images_to_compare.append(image)
    for image_to_compare, title in zip(all_images_to_compare, titles):
        # 1) Check if 2 images are equals
        if original.shape == image_to_compare.shape:
            print("The images have same size and channels")
            difference = cv2.subtract(original, image_to_compare)
            b, g, r = cv2.split(difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                print("Similarity: 100% (equal size and channels)")
                #break
        # 2) Check for similarities between the 2 images
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        matches = flann.knnMatch(desc_1, desc_2, k=2)
        good_points = []
        for m, n in matches:
            if m.distance > 0.6*n.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(kp_1) >= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)
        print("Title: " + title)
        percentage_similarity = len(good_points) / number_keypoints * 100
        resultados.append( int(percentage_similarity))
        print("Similarity: " + str(int(percentage_similarity)) + "\n")

    tkinter.messagebox.showinfo(title="Alerta",message="Termino buscar Coincidencias ")
    print(resultados)
    idx,val_m = mayor(resultados)
    
    #ids,person,porcentaje = mayor(resultados)
    #mayor(resultados)
    lblPorcentaje.config(text = " Se encontro un "+str(val_m )+" %")
    agregar_foto(persona[idx])
    datos_preparet(idx)
    vaciar()
def vaciar():
    global resultados
    resultados = []
ida = 0
def datos_preparet(position):
    global ida
    ida = position
#print(resultados)
#buscar el mayor de todas las coincidencias 
def mayor(lista):
    val_max = lista[0];
    ids = 0
    for idx,v in enumerate(lista):
        print (str(v)+" => "+str(val_max))
        if (v>val_max):
            val_max = v
            ids = idx
            print ("indice %s valor %s" %(idx,v))
    print(val_max)
    print(ids)
    return ids,val_max
#frame ver datos ventana secundaria
def agregar_foto(path_image):
    
    # Leer la imagen de entrada y la redimensionamos
    global img_rostros
    print(img_rostros+path_image)
    image = cv2.imread(img_rostros+path_image)
    image= imutils.resize(image, height=380)
    # Para visualizar la imagen de entrada en la GUI
    imageToShow= imutils.resize(image, width=180)
    imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(imageToShow )
    img = ImageTk.PhotoImage(image=im)
    lblMatch.configure(image=img)
    lblMatch.image = img

def similitudes():
    global ida
    imagen = Image.open("huella.bmp").convert('L')
    imagen = imagen.resize((550,550))
    imagen2 = Image.open(titulos[ida]).convert('L')
    imagen2 = imagen2.resize((550,550))
    #guardando las imagenes
    imagen.save("ima1.jpg")
    imagen2.save("ima2.jpg")

    #proceso de imagenes
    img1 = cv2.imread('ima1.jpg',cv2.IMREAD_GRAYSCALE) # queryImage
    img2 = cv2.imread('ima2.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    #print(len(matchesMask))

    salto = int(len(matchesMask)/15)

    # ratio test as per Lowe's paper
    #print(matches
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches[::salto],None,flags=2)
    plt.title("Similitudes Dactilares Encontradas", 
          fontdict={'family': 'serif', 
                    'color' : 'darkblue',
                    'weight': 'bold',
                    'size': 18})
    plt.imshow(img3,),plt.show()
#cargar datos al slider

def envia_boton():
   ventana_nueva1 = Toplevel()
   ventana_nueva1.geometry("400x200")
   ventana_nueva1.title("Ventana secundaria")
   etiqueta = Label(ventana_nueva1,
                    text="El valor introducido en la ventana principal es: ").grid(row=0)
#///////////////////////////
#/////// ver las similitudes entre las huellas
def ver_datos():
    imagen = Image.open("images/imageProcesada.jpg")
    imagen = imagen.resize((550,550))

    imagen2 = Image.open("images/cue2.BMP")
    imagen2 = imagen2.resize((550,550))
    #guardando las imagenes
    imagen.save("ima1.jpg")
    imagen2.save("ima2.jpg")
    #proceso de imagenes
    img1 = cv2.imread('ima1.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv2.imread('ima2.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

#///////////////////// similitudes entre las huellas ////////////

img_rostros = 'personas/'
with os.scandir(img_rostros) as ficheros:
    for fichero in ficheros:
        persona.append(fichero.name)
#///////////////////////////

root = Tk()
root.title("Reconocimiento Dactilar")
root.geometry("1000x500")
root.resizable(width=True, height=True)
#elemtos menu
barra_menus = tk.Menu()
# Crear el primer menú.
menu_archivo = Menu(barra_menus, tearoff=False)
# Agregarlo a la barra.
barra_menus.add_command(label="Nuevo Usuario")
barra_menus.add_command(label="Eliminar Usuario")
root.config(menu=barra_menus)
#rostro a importar
InputImageRostro = Label(root,bg="black")
InputImageRostro.place(x=470,y=80)
InputImage = Label(root,bg="black")
InputImage.place(x=740,y=80)
#frame registro
#no borrar es importante xd

capturar = Frame(root)
capturar.pack()
capturar.config(bg="cadetblue")
capturar.config(width=1000,height=500)

#boton 
# Creamos el botón para elegir la imagen de entrada
btn = Button(capturar, text="Elegir imagen", width=15, command=elegir_imagen,font=("Calisto MT",12,"bold"))
btn.place(x=50,y=5)
#boton para procesar imagen
lblInputImage = Label(capturar)
lblInputImage.place(x=50,y=40)

#boton 
# Creamos el botón para procesar Imagen
#boton capturar
botonBuscar = tk.Button(capturar, text="Tratar Imagen",width=15,command=guardar_huella,font=("Calisto MT",12,"bold"))
botonBuscar.place(x=60,y=300)
#////////////////////////////
porcentaje = 0
#donde se pondra el porcentaje de similituc
lblPorcentaje = Label(capturar,text=" Se encontro un "+str(porcentaje)+" %")
lblPorcentaje .place(x=550,y=40)
#donde se pone imagen de coincidedncias
lblMatch = Label(capturar)
lblMatch.place(x=350,y=40)
#boton 
# Creamos el botón para procesar Imagen
#boton capturar
botonBuscar = tk.Button(capturar, text="Buscar Coincidecias",cursor = "hand2",width=15,command=buscar_coincidencias, height=2,font=("Calisto MT",12,"bold"))
botonBuscar.place(x=100,y=400)

#boton capturar
botonBuscar = tk.Button(capturar, text="Ver Datos",cursor = "hand2",width=15,command=similitudes, height=2,font=("Calisto MT",12,"bold"))
botonBuscar.place(x=550,y=100)

#////////////////////////////

root.mainloop()
