# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:06:42 2018

@author: pedzenon
"""

import io
import os
import numpy as np
import json
from os import listdir

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.protobuf.json_format import MessageToJson

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate

fileDir = os.path.dirname(os.path.realpath('__file__'))
fileOut= os.path.join(fileDir, 'imagenes')

# levanto todos los files, menos los que se rompieron
nombres_files = [f for f in listdir(fileOut)
                         if (((".jpg" in f) | (".png" in f)) & (f[0] != ".") & (f[0] != "~")  & ("roto" not in f) )]

imagen_codes = [x.split('.')[0] for x in nombres_files  ]  # nombre del codigo del archivo
extension_codes = [x.split('.')[1] for x in nombres_files  ] # .jpg o .png


def generate_visionData(nombres_file,imagen_code):
    
    # abro la imagen
    file_name = os.path.join(
        fileOut,
        nombres_file)
    
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    
    image = types.Image(content=content)
    
    # Performs label detection on the image file
    
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    # genero el json
    output = [{}]
    for text in texts:
        output += [json.loads(MessageToJson(text))]
    output.pop(0)
    
    path_out = fileOut + '\\' + imagen_code+ '_texts'+ '.json'
    
    with open(
             path_out, 'w') as outfile:
            json.dump(output, outfile)
            
    #for text in texts:
    #    print(text.description)
    #    
    #vertices = ([(vertex.x, vertex.y)
    #                for vertex in texts[0].bounding_poly.vertices])
    
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    output = [{}]
    for label in labels:
        output += [json.loads(MessageToJson(label))]
    output.pop(0)
    
    path_out = fileOut + '\\' + imagen_code+ '_labels' +'.json'
    
    with open(
             path_out, 'w') as outfile:
            json.dump(output, outfile)
            
    #for label in labels:
    #    print(label.description)
    
    response = client.face_detection(image=image)
    faces = response.face_annotations
    
    output = [{}]
    for face in faces:
        output += [json.loads(MessageToJson(face))]
    output.pop(0)
    
    path_out = fileOut + '\\' + imagen_code+ '_faces' +'.json'
    
    with open(
             path_out, 'w') as outfile:
            json.dump(output, outfile)
            
# cargo la data
for i in range(len(imagen_codes)):
    print("cargando: ",nombres_files[i])
    generate_visionData(nombres_files[i],imagen_codes[i])
    print("generados json de: ",nombres_files[i])
