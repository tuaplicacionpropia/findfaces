#!/usr/bin/env python2.7
#coding:utf-8

import os
import numpy as np
import argparse
import cv2
import screeninfo
import sys
import math

class FacesTools:

  def __init__ (self):
    pass

  def cropAllFaces (self, imgPath):
    faces = self.detect(imgPath, confidence=0.5)
    cwd = os.getcwd()
    if imgPath is not None and not os.path.isabs(imgPath):
      imgPath = os.path.join(cwd, imgPath)
    image = cv2.imread(imgPath)
    for face in faces:
      #startY and endY coordinates, followed by the startX and endX 
      startY = face['start'][1]
      endY = face['end'][1]
      startX = face['start'][0]
      endX = face['end'][0]
      cropped = image[startY:endY, startX:endX]
      cv2.imshow("cropped" + face['label'], cropped)
    cv2.waitKey(0)

  def detect (self, imgPath, confidence=0.5):
    result = None
    cwd = os.getcwd()
    if imgPath is not None and not os.path.isabs(imgPath):
      imgPath = os.path.join(cwd, imgPath)
    confidence=float(confidence)

    script_path = os.path.dirname(os.path.abspath(__file__))

    prototxt = os.path.join(os.path.join(script_path, 'models'), 'deploy.prototxt.txt')
    model = os.path.join(os.path.join(script_path, 'models'), 'res10_300x300_ssd_iter_140000.caffemodel')

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    #size = 300
    #size = 1024
    #size = 2048
    #size = 4096
    scaleSizes = [4096, 2048, 4096, 4096, 2048, 2048, 1024, 300]
    sizes = [4096, 2048, 4096/5, 4096/10, 2048/5, 2048/10, 300, 300]
    #scaleSizes = [4096, 2048]
    #sizes = [4096/5, 2048]
    #scaleSizes = [4096, 300]
    #sizes = [4096 / 10, 300]
    #sizes = [300]
    #blue, green, red, yellow
    colors = [(128, 0, 128), (0, 128, 128), (64, 28, 128), (64, 28, 128), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]

    foundList = list()

    imgWidth = None
    imgHeight = None

    labelIdx = 1

    idx = 0
    for scaleSize in scaleSizes:
      image = cv2.imread(imgPath)
      (h, w) = image.shape[:2]
      if imgWidth is None and imgHeight is None:
        imgWidth = w
        imgHeight = h

      size = sizes[idx]
      color = colors[idx]
      idx += 1

      #blob = cv2.dnn.blobFromImage(cv2.resize(image, (size, size)), 1.0, (size, size), (103.93, 116.77, 123.68))
      scaleFactor = 1.0
      #scaleFactor = float(h) / float(w)
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (scaleSize, scaleSize)), scaleFactor, (size, size), (103.93, 116.77, 123.68), swapRB=True)

      net.setInput(blob)
      detections = net.forward()

      for i in range(0, detections.shape[2]):
        curConfidence = detections[0, 0, i, 2]

        if curConfidence > confidence:
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          text = "{:.2f}%".format(curConfidence * 100)
          y = startY - 10 if startY - 10 > 10 else startY + 10
          adaptScaleX = float(w) / float(scaleSize)
          adaptScaleY = float(h) / float(scaleSize)
          print("width = " + str(w) + " height = " + str(h) + " scaleSize = " + str(scaleSize) + " adaptScaleX = " + str(adaptScaleX) + " adaptScaleY = " + str(adaptScaleY))
          foundList.append({'start': (startX, startY), 'end': (endX, endY), 'confidence': curConfidence, 'label': 'label_' + str(labelIdx)})
          labelIdx += 1
          if False:
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.95, color, 2)

      if False:
        W = float((screeninfo.get_monitors()[0]).width)*0.5
        height, width, depth = image.shape
        imgScale = W/width
        newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
        newimg = cv2.resize(image,(int(newX),int(newY)))
        #cv2.imwrite("resizeimg.jpg",newimg)

        # show the output image
        cv2.imshow("Output_" + str(scaleSize) + "-" + str(size), newimg)

    #print("foundList")
    #for i in range(0, len(foundList)):
    #  print("" + str(i) + ". " + str(foundList[i]))
    #print(str(foundList))
    #for i in range(0, 100):
    #  print("IMG WIDTH = " + str(imgWidth) + " IMG HEIGHT = " + str(imgHeight))
    foundList = self.removeBadFaces(foundList, imgWidth, imgHeight)
    #for i in range(0, 100):
    #  print("START FACES")
    #for i in range(0, len(foundList)):
    #  print("" + str(i) + ". " + str(foundList[i]))
    faces = self.selectFaces(foundList)
    faces = self.selectSimilarSurfaceFaces(faces)
    faces = self.discardDistantes(faces)
    #for i in range(0, 100):
    #  print("faces")
    #print(str(faces))
    if False:
      image = cv2.imread(imgPath)

      for face in faces:
        startX = face['start'][0]
        startY = face['start'][1]
        endX = face['end'][0]
        endY = face['end'][1]
        faceConfidence = face['confidence']
        text = "{:.2f}%".format(faceConfidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        color = (128, 128, 128)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.95, color, 2)
        
        W = float((screeninfo.get_monitors()[0]).width)*0.5
        height, width, depth = image.shape
        imgScale = W/width
        newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
        newimg = cv2.resize(image,(int(newX),int(newY)))
        
        cv2.imshow("REAL_IMAGE", newimg)

    #cv2.waitKey(0)
    #print("faces")
    #for i in range(0, len(faces)):
    #  print("" + str(i) + ". " + str(faces[i]))
    result = faces
    return result

  def detectBak20190413 (self, imgPath, confidence=0.5):
    cwd = os.getcwd()
    if imgPath is not None and not os.path.isabs(imgPath):
      imgPath = os.path.join(cwd, imgPath)
    confidence=float(confidence)

    script_path = os.path.dirname(os.path.abspath(__file__))

    prototxt = os.path.join(os.path.join(script_path, 'models'), 'deploy.prototxt.txt')
    model = os.path.join(os.path.join(script_path, 'models'), 'res10_300x300_ssd_iter_140000.caffemodel')

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    #size = 300
    #size = 1024
    #size = 2048
    #size = 4096
    scaleSizes = [4096, 2048, 4096, 4096, 2048, 2048, 1024, 300]
    sizes = [4096, 2048, 4096/5, 4096/10, 2048/5, 2048/10, 300, 300]
    #scaleSizes = [4096, 2048]
    #sizes = [4096/5, 2048]
    #scaleSizes = [4096, 300]
    #sizes = [4096 / 10, 300]
    #sizes = [300]
    #blue, green, red, yellow
    colors = [(128, 0, 128), (0, 128, 128), (64, 28, 128), (64, 28, 128), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]

    foundList = list()

    imgWidth = None
    imgHeight = None

    labelIdx = 1

    idx = 0
    for scaleSize in scaleSizes:
      image = cv2.imread(imgPath)
      (h, w) = image.shape[:2]
      if imgWidth is None and imgHeight is None:
        imgWidth = w
        imgHeight = h

      size = sizes[idx]
      color = colors[idx]
      idx += 1

      #blob = cv2.dnn.blobFromImage(cv2.resize(image, (size, size)), 1.0, (size, size), (103.93, 116.77, 123.68))
      scaleFactor = 1.0
      #scaleFactor = float(h) / float(w)
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (scaleSize, scaleSize)), scaleFactor, (size, size), (103.93, 116.77, 123.68), swapRB=True)

      net.setInput(blob)
      detections = net.forward()

      for i in range(0, detections.shape[2]):
        curConfidence = detections[0, 0, i, 2]

        if curConfidence > confidence:
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          text = "{:.2f}%".format(curConfidence * 100)
          y = startY - 10 if startY - 10 > 10 else startY + 10
          adaptScaleX = float(w) / float(scaleSize)
          adaptScaleY = float(h) / float(scaleSize)
          print("width = " + str(w) + " height = " + str(h) + " scaleSize = " + str(scaleSize) + " adaptScaleX = " + str(adaptScaleX) + " adaptScaleY = " + str(adaptScaleY))
          foundList.append({'start': (startX, startY), 'end': (endX, endY), 'confidence': curConfidence, 'label': 'label_' + str(labelIdx)})
          labelIdx += 1
          cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
          cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.95, color, 2)

      W = float((screeninfo.get_monitors()[0]).width)*0.5
      height, width, depth = image.shape
      imgScale = W/width
      newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
      newimg = cv2.resize(image,(int(newX),int(newY)))
      #cv2.imwrite("resizeimg.jpg",newimg)

      # show the output image
      cv2.imshow("Output_" + str(scaleSize) + "-" + str(size), newimg)

    print("foundList")
    for i in range(0, len(foundList)):
      print("" + str(i) + ". " + str(foundList[i]))
    #print(str(foundList))
    for i in range(0, 100):
      print("IMG WIDTH = " + str(imgWidth) + " IMG HEIGHT = " + str(imgHeight))
    foundList = self.removeBadFaces(foundList, imgWidth, imgHeight)
    for i in range(0, 100):
      print("START FACES")
    for i in range(0, len(foundList)):
      print("" + str(i) + ". " + str(foundList[i]))
    faces = self.selectFaces(foundList)
    faces = self.selectSimilarSurfaceFaces(faces)
    faces = self.discardDistantes(faces)
    for i in range(0, 100):
      print("faces")
    print(str(faces))
    if True:
      image = cv2.imread(imgPath)

      for face in faces:
        startX = face['start'][0]
        startY = face['start'][1]
        endX = face['end'][0]
        endY = face['end'][1]
        faceConfidence = face['confidence']
        text = "{:.2f}%".format(faceConfidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        color = (128, 128, 128)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.95, color, 2)
        
        W = float((screeninfo.get_monitors()[0]).width)*0.5
        height, width, depth = image.shape
        imgScale = W/width
        newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
        newimg = cv2.resize(image,(int(newX),int(newY)))
        
        
        cv2.imshow("REAL_IMAGE", newimg)

    cv2.waitKey(0)
    print("faces")
    for i in range(0, len(faces)):
      print("" + str(i) + ". " + str(faces[i]))

  def loadLimitConfidence (self, faces):
    result = None
    
    sumConfidence = 0.0
    for face in faces:
      sumConfidence += face['confidence']
    limit = sumConfidence / (len(faces))
    
    limit = min(limit, 1.65)
    
    result = max(0.8, limit)
    result = 0.8 if limit < 1.0 else limit
    #result = 0.8
    return result

  def discardDistantes (self, faces):
    result = list()
    limitConfidence = self.loadLimitConfidence(faces)
    for face in faces:
      if face['confidence'] > limitConfidence:
        result.append(face)
      else:
        print("DESCARTADO POR NO ALCANZAR EL LÍMITE DE CONFIANZA " + str(limitConfidence) + " face = " + str(face))
    return result

  def selectMostConfidenceFace (self, faces):
    result = None
    maxConfidence = 0.0
    for face in faces:
      value = self.getSurfaceRectangle(face)
      #value = 1.0
      value *= face['confidence']
      if value > maxConfidence:
        result = face
        maxConfidence = value
    return result

  def selectSimilarSurfaceFaces (self, faces):
    result = list()
    limitSurface = 150.0
    mostConfidenceFace = self.selectMostConfidenceFace(faces)
    for face in faces:
      diffSurface = self.getDistanceSurface(face, mostConfidenceFace)
      if diffSurface < limitSurface:
        result.append(face)
      else:
        print "SURFACE MOST = " + str(self.getSurfaceRectangle(mostConfidenceFace))
        print "SURFACE DESCARTADA = " + str(self.getSurfaceRectangle(face))
        print "diffSurface = " + str(diffSurface)
        print "MOST SURFACE FACES " + str(mostConfidenceFace)
        print "DESCARTADA POR SIMILAR SURFACE FACES " + str(face)
    return result

  def removeBadFaces (self, faces, imgWidth, imgHeight):
    result = list()
    for face in faces:
      faceWidth = self.getWidthRectangle(face)
      faceHeight = self.getHeightRectangle(face)
      valid = True
      valid = valid and face['start'][0] > 0 and face['start'][0] < imgWidth
      valid = valid and face['start'][1] > 0 and face['start'][1] < imgHeight
      valid = valid and face['end'][0] > 0 and face['end'][0] < imgWidth
      valid = valid and face['end'][1] > 0 and face['end'][1] < imgHeight
      if valid:
        result.append(face)
      else:
        print "DESCARTANDO POR BAD FACES " + str(face)
    return result

  def selectFaces (self, foundList):
    result = list()
    for face in foundList:
      groupIdx = 0
      foundGroup = None
      for group in result:
        print(">>>>>>COMPARANDO " + str(face) + " CON " + str(group))
        if self.checkSimilarFace(face, group):
          foundGroup = group
          break
        groupIdx += 1
      if foundGroup is None:
        print(">>>>>>NADA SIMILAR")
        result.append(face)
      else:
        if face['confidence'] > foundGroup['confidence']:
          print ">>>>>>DESCARTANDO POR PARECIDO GROUP " + str(foundGroup)
          face['confidence'] += foundGroup['confidence']
          result[groupIdx] = face
        else:
          foundGroup['confidence'] += face['confidence']
          print ">>>>>>DESCARTANDO POR PARECIDO FACE " + str(face)
          
        
    return result

  def getSurfaceRectangle (self, face):
    result = 0
    w = face['end'][0] - face['start'][0]
    h = face['end'][1] - face['start'][1]
    result = w * h
    return result

  def getWidthRectangle (self, face):
    result = 0
    result = face['end'][0] - face['start'][0]
    return result

  def getHeightRectangle (self, face):
    result = 0
    result = face['end'][1] - face['start'][1]
    return result

  def getDistanceFace (self, face1, face2):
    result = 0.0
    pow1 = pow(face2['start'][0] - face1['start'][0], 2)
    pow2 = pow(face2['start'][1] - face1['start'][1], 2)
    result = math.sqrt(pow1 + pow2)
    return result

  def getDistanceSurface (self, face1, face2):
    result = 0.0
    surface1 = self.getSurfaceRectangle(face1)
    surface2 = self.getSurfaceRectangle(face2)
    result = ((float(max(surface1, surface2)) / float(min(surface1, surface2))) - 1.0) * 100.0
    return result

  def checkContainsPoint (self, face1, point):
    result = True
    result = result and face1['start'][0] <= point[0] and face1['end'][0] >= point[0]
    result = result and face1['start'][1] <= point[1] and face1['end'][1] >= point[1]
    print "comparing overlapping face " + str(face1) + " with point " + str(point)
    return result

  def checkOverlappedRectangleBase (self, face1, face2):
    result = False
    print "comparing overlapped " + str(face1) + " with " + str(face2)
    result = result or self.checkContainsPoint(face1, face2['start'])
    result = result or self.checkContainsPoint(face1, [face2['start'][0], face2['end'][1]])
    result = result or self.checkContainsPoint(face1, [face2['end'][0], face2['start'][1]])
    result = result or self.checkContainsPoint(face1, face2['end'])
    result = result or self.checkContainsPoint(face1, [face2['start'][0] + ((face2['end'][0] - face2['start'][0])/2), face2['start'][1] + ((face2['end'][1] - face2['start'][1])/2)])
    return result


  def checkOverlappedRectangle (self, face1, face2):
    result = False
    result = result or self.checkOverlappedRectangleBase(face1, face2)
    result = result or self.checkOverlappedRectangleBase(face2, face1)
    return result

  def checkSimilarFace (self, face1, face2):
    result = True
    limitSurface = 100.0

    #limitDistanceW = min(self.getWidthRectangle(face1), self.getWidthRectangle(face2))
    #limitDistanceH = min(self.getHeightRectangle(face1), self.getHeightRectangle(face2))
    #limitDistance = min(limitDistanceW, limitDistanceH)
    #limitDistance = 1

    diffSurface = self.getDistanceSurface(face1, face2)
    #distance = self.getDistanceFace(face1, face2)

    result = result and diffSurface < limitSurface
    result = result and self.checkOverlappedRectangle(face1, face2)
    #result = result and distance < limitDistance

    
    print("face1 = " + str(face1))
    print("face2 = " + str(face2))
    print("diffSurface = " + str(diffSurface))
    #print("distance = " + str(distance))
    print("limitSurface = " + str(limitSurface))
    #print("limitDistance = " + str(limitDistance))
    print("SIMILAR????? = " + str(result))

    return result

  

  def checkSimilarFace_old (self, face1, face2):
    result = True
    limitSurface = 100.0

    limitDistanceW = min(self.getWidthRectangle(face1), self.getWidthRectangle(face2))
    limitDistanceH = min(self.getHeightRectangle(face1), self.getHeightRectangle(face2))
    limitDistance = min(limitDistanceW, limitDistanceH)
    #limitDistance = 1

    diffSurface = self.getDistanceSurface(face1, face2)
    distance = self.getDistanceFace(face1, face2)

    result = result and diffSurface < limitSurface
    result = result and distance < limitDistance

    
    print("face1 = " + str(face1))
    print("face2 = " + str(face2))
    print("diffSurface = " + str(diffSurface))
    print("distance = " + str(distance))
    print("limitSurface = " + str(limitSurface))
    print("limitDistance = " + str(limitDistance))
    print("SIMILAR????? = " + str(result))

    return result

  def detect2 (self, imgPath, confidence=0.5):
    cwd = os.getcwd()
    if imgPath is not None and not os.path.isabs(imgPath):
      imgPath = os.path.join(cwd, imgPath)
    confidence=float(confidence)

    script_path = os.path.dirname(os.path.abspath(__file__))

    prototxt = os.path.join(os.path.join(script_path, 'models'), 'deploy.prototxt.txt')
    model = os.path.join(os.path.join(script_path, 'models'), 'res10_300x300_ssd_iter_140000.caffemodel')

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    #size = 300
    #size = 1024
    #size = 2048
    #size = 4096
    scaleSizes = [4096, 2048, 4096, 4096, 2048, 2048, 1024, 300]
    sizes = [4096, 2048, 4096/5, 4096/10, 2048/5, 2048/10, 300, 300]
    #sizes = [300]
    #blue, green, red, yellow
    colors = [(128, 0, 128), (0, 128, 128), (64, 28, 128), (64, 28, 128), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    idx = 0
    for scaleSize in scaleSizes:
      image = cv2.imread(imgPath)
      (h, w) = image.shape[:2]

      size = sizes[idx]
      color = colors[idx]
      idx += 1

      #blob = cv2.dnn.blobFromImage(cv2.resize(image, (size, size)), 1.0, (size, size), (103.93, 116.77, 123.68))
      scaleFactor = 1.0
      #scaleFactor = float(h) / float(w)
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (scaleSize, scaleSize)), scaleFactor, (size, size), (103.93, 116.77, 123.68), swapRB=True)

      net.setInput(blob)
      detections = net.forward()

      for i in range(0, detections.shape[2]):
        curConfidence = detections[0, 0, i, 2]

        if curConfidence > confidence:
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          text = "{:.2f}%".format(curConfidence * 100)
          y = startY - 10 if startY - 10 > 10 else startY + 10
          cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
          cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.95, color, 2)

      W = float((screeninfo.get_monitors()[0]).width)*0.5
      height, width, depth = image.shape
      imgScale = W/width
      newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
      newimg = cv2.resize(image,(int(newX),int(newY)))
      #cv2.imwrite("resizeimg.jpg",newimg)

      # show the output image
      cv2.imshow("Output_" + str(scaleSize) + "-" + str(size), newimg)

    cv2.waitKey(0)

if True and __name__ == '__main__':
  for m in screeninfo.get_monitors():
    print(">width = " + str(m.width))
    print(">height = " + str(m.height))
  tools = FacesTools()
  imgPath = sys.argv[1] if len(sys.argv) > 1 else "/media/jmramoss/ALMACEN/pypi/findfaces/findfaces/images/img07.jpg"
  confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
  tools.detect(imgPath, confidence)
  print(str(sys.argv))
