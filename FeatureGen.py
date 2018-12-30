# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:37:53 2018

@author: pedzenon
"""

import cv2
import io
import os
import numpy as np
import json
from os import listdir
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import operator
from scipy.stats import mode
import pandas as pd

# @fn get_imgProp: extrae los colores de la imagen, el ancho, alto y size

def get_imgProp(imagen):    
    
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)  # transforma al campo HSV
    h,s,v = cv2.split(hsv)  # me quedo con cada compoente
    
    #size = h[h>0].shape[0]
    size = h.shape[0]*h.shape[1]
    height = h.shape[0]
    width = h.shape[1]
    
    brillo = np.mean(v)/255
    sat = np.mean(s)/255
    
    if(width >= height):
        ratio = [width/height,1]
    else:
        ratio = [1,height/width]
        
    
    #para transformar de s_real a s_opencv:
    # 5.603    +   sreal* 2.486 
    
    #para transformar de v_real a v_opencv: 
    #-0.4353   +   v_real*  2.5559  = v_opencv
    
    # umbral minimo y maximo de hue
    colores = {'celeste' : (81,104),'azul':(105,129),
               'rosa':(130,170),'rojo1':(171,179),'rojo2':(1,6),'naranja':(7,17),'amarillo':(18,35),
              'verde':(36,80)}
    
    # valor umbral de v
    negro_v = 54  # <= 30
    
    # valor umbral de v
    blanco_v = 230  # >= 240
    #valor umbral de s
    blanco_s = 10   # <= 10
    
    gris_v = (46,183)
    gris_s = 30 # <= 30
    
    cremita_s = (blanco_s,47)
    cremita_v = 220
    
    mask_generic = cv2.inRange(hsv, np.array([0,40,50]),
                                             np.array([255,255,255]))
    res = cv2.bitwise_and(hsv,hsv, mask= mask_generic)
    h_trunc,s_trunc,v_trunc = cv2.split(res)
    count,bins = np.histogram(h_trunc.ravel(),256,[0,256])
    #hist(s.ravel(),256,[0,256]);plt.plot()
    celeste = sum(count[colores["celeste"][0]:colores["celeste"][1]])/size * 100
    azul = sum(count[colores["azul"][0]:colores["azul"][1]])/size * 100
    rosa = sum(count[colores["rosa"][0]:colores["rosa"][1]])/size * 100
    rojo1 = sum(count[colores["rojo1"][0]:colores["rojo1"][1]])
    rojo2 = sum(count[colores["rojo2"][0]:colores["rojo2"][1]])
    rojo = (rojo1+rojo2)/size * 100
    naranja = sum(count[colores["naranja"][0]:colores["naranja"][1]])/size * 100
    amarillo = sum(count[colores["amarillo"][0]:colores["amarillo"][1]])/size * 100
    verde = sum(count[colores["verde"][0]:colores["verde"][1]])/size * 100
    
    #negro = sum((v<negro_v) & (v>0) )/size * 100
    negro = sum((v<negro_v) )/size * 100
    blanco = sum((v>blanco_v) & (s < blanco_s))/size * 100
    gris = sum((v>gris_v[0]) & (v<gris_v[1]) & (s < gris_s))/size * 100
    cremita = sum((s>cremita_s[0]) & (s<cremita_s[1]) & (v > cremita_v))/size * 100

    return {"celeste":celeste,"azul":azul,"rosa":rosa,"rojo":rojo,"naranja":naranja,
            "amarillo":amarillo,"verde":verde,"negro":negro,"blanco":blanco,"cremita":cremita,
            "gris":gris,"size":size,"height":height,"width":width,"brillo":brillo,"saturacion":sat,
            "ratio1":ratio[0], "ratio2":ratio[1] }    


#################################################################################################

## FACES:


def face_area(face):
    if("x" not in face[0].keys()):
        face[0]["x"] = 0
    
    if("y" not in face[0].keys()):
        face[0]["y"] = 0
        
    return (face[1]["x"] - face[0]["x"], face[2]["y"] - face[0]["y"] )

def imagen_faces(img,fileDir,fileOut,size):
    
    with open(fileOut + '\\'+ img + '_faces.json') as data_file:
        faces = json.load(data_file)
    
    n_faces =  len(faces)
    
    if(n_faces > 0):
        sentiment = []
        num_sentiment = {'UNKNOWN':0, 'VERY_UNLIKELY':1, 'UNLIKELY':2, 'POSSIBLE':3,
                               'LIKELY':4, 'VERY_LIKELY':5}                    
        for face in faces:
            google_vision = {}
            google_vision['3anger'] = num_sentiment[face["angerLikelihood"]]
            google_vision['2surprise'] = num_sentiment[face["surpriseLikelihood"]]
            google_vision['1joy'] = num_sentiment[face["joyLikelihood"]]        
            google_vision['4sorrow'] = num_sentiment[face["sorrowLikelihood"]]
        
            sentiment_aux = max(google_vision.items(), key=operator.itemgetter(1))[0] 
            
            if(google_vision[sentiment_aux] > 2):       
                sentiment.append(sentiment_aux)
        
        sentiment = sorted(sentiment)
        
        if(len(sentiment)):
            sentiment_ganador = mode(sentiment)[0][0]    
            sentiment_ganador = sentiment_ganador[1:] 
        else:
            sentiment_ganador = ""
        
        # area que ocupa la imagen
        area_faces = 0
        for face in faces:
            xLen_face,yLen_face = face_area(face["boundingPoly"]["vertices"])
            
            area_faces += np.abs(xLen_face*yLen_face)
            
        area_faces = 100*area_faces/size
    
    else:
        area_faces = 0
        sentiment_ganador = ""
            
    return {"n_faces":n_faces, "sentiment":sentiment_ganador, "area_faces":area_faces }

## TEXTO:

def text_area(text):
    if("x" not in text["boundingPoly"]["vertices"][0].keys()):
        return 0
    
    if("y" not in text["boundingPoly"]["vertices"][1].keys()):
        return 0
    
    if("y" not in text["boundingPoly"]["vertices"][2].keys()):
        return 0
    
    if("x" not in text["boundingPoly"]["vertices"][1].keys()):
        return 0
        
    return (np.abs(text["boundingPoly"]["vertices"][1]["x"] - text["boundingPoly"]["vertices"][0]["x"]) * (text["boundingPoly"]["vertices"][2]["y"] - text["boundingPoly"]["vertices"][1]["y"]))

def imagen_text(img,fileDir,fileOut,marcas,size):
    
    with open(fileOut + '\\'+ img + '_texts.json') as data_file:
        texts = json.load(data_file)
    
    area_text = 0
    
    if (len(texts)):  # en caso que no diga nada el anuncio
        full_text = texts.pop(0)  # descarto el primero que es una descripcion del texto
        
        # me fjo si aparece el nombre de la marca en el copy y el area que ocupa en la imagen
        ismarca = 0
        for text in texts:
            isIn = [1 if(x in text["description"].lower()) else 0 for x in marcas]
            
            if(sum(isIn) > 0):
                ismarca += 1
                
            area_text += text_area(text)

        area_text = 100*area_text/size
        
        # cantidad de palabras
        num_text_leters = len( [x for x in full_text["description"] if(("\n" not in x) &
                            (" " not in x) )])
    else:
        ismarca = 0
        num_text_leters = 0
        area_text = 0
        
    return {"ismarca":ismarca, "area_text":area_text, "num_text_leters":num_text_leters }

# LABELS:
    
def getUniqueItems(iterable):
    result = []
    for item in iterable:
        if item not in result:
            result.append(item)
    return result


def imagen_labels(fileOut,file):
    
    stop_words = ["brand","advertising","and"]
    
    with open(fileOut + '\\'+ file + '_labels.json') as data_file:
        labels = json.load(data_file)    
    
    aux = []  
    for label in labels:
        aux = aux + label["description"].split()
        
    aux =  getUniqueItems(aux)   
        
    all_labels = [x for x in aux if(x not in stop_words )]  # saco stop words
    
    if((len(all_labels) == 1) & ("product" in all_labels)):
        soloProducto = 1
    else:
        soloProducto = 0
    
    return {'labels':all_labels,'soloProducto':soloProducto}

#################################################################################################################
#############################################    Main    ######################################################
#################################################################################################################

## LABELS: Hacer analisis de bag of words

fileDir = os.path.dirname(os.path.realpath('__file__'))
fileOut= os.path.join(fileDir, 'imagenes')

# levanto archivos ya cargados
nombres_files = [f for f in listdir(fileOut)
                         if (("labels.json" in f) & (f[0] != ".") & (f[0] != "~"))]

stop_words = ["brand","advertising","and","yellow","magenta","red","blue","pink","purple","text","violet","photo"]

all_labels = []  

for file in nombres_files:
    with open(fileOut + '\\'+ file) as data_file:
        labels = json.load(data_file)    
    aux = []  
    for label in labels:
        aux = aux + label["description"].split()
        
    aux =  getUniqueItems(aux)   
    all_labels = all_labels + aux    
    
all_labels = [x for x in all_labels if(x not in stop_words )]  # saco stop words

#########################################################################################################♣

data = pd.read_excel("ImageWithKPI.xlsx")

fileDir = os.path.dirname(os.path.realpath('__file__'))
fileOut= os.path.join(fileDir, 'imagenes')

# levanto todos los files
nombres_files = [f for f in listdir(fileOut)
                         if (((".jpg" in f) | (".png" in f) | (".jpeg" in f)) & (f[0] != ".") & (f[0] != "~")  & ("roto" not in f) )]

ids = [x.split('.')[0] for x in nombres_files]

nombres_jsons = [f for f in listdir(fileOut)
                         if ((".json" in f) & (f[0] != ".") & (f[0] != "~")  & ("roto" not in f) )]

marcas = ["PRIVATE"]


#all_data = pd.DataFrame([],columns = cols)
l_all_data = []
                        
for file in nombres_files:
    imagen = cv2.imread(fileOut + '\\'+ file )
    size =  np.size(imagen, 0)*np.size(imagen, 1)
    columns1 = get_imgProp(imagen)
    file_imgCode = file.split('.')[0]
    columns2 = imagen_faces(file_imgCode,fileDir,fileOut,size)
    print(file_imgCode)
    columns3 = imagen_text(file_imgCode,fileDir,fileOut,marcas,size)
    columns4 = imagen_labels(fileOut,file_imgCode)
    aux = columns1
    aux.update(columns2)
    aux.update(columns3)
    aux.update(columns4)
    aux.update({"IMGcode":file_imgCode})
    l_all_data.append(aux)
#    aux = pd.DataFrame([aux])[cols]
#    all_data = all_data.append(aux,ignore_index=True)

features = pd.DataFrame(l_all_data)
features["IMGcode"] = pd.to_numeric(features["IMGcode"])
features = pd.merge(features,data,on = ["IMGcode"])

features = features.dropna(subset = ['Like', 'Love','Haha', 'Angry', 'Wow', 'Sad', 'Comment', 'Share'])
features["engagement"] = list(features[['Like', 'Love','Haha', 'Angry', 'Wow', 'Sad', 'Comment', 'Share']].sum(axis = 1))

# lo paso a excel para analizar en el descriptivo!!
features.to_excel("ToAnalize.xlsx")
