#!/usr/bin/env python2.7
#coding:utf-8

import os
import numpy as np
import argparse
import cv2
import screeninfo
import sys

class FacesTools:

  def __init__ (self):
    pass

  def detect (self, imgPath, confidence=0.8):
    cwd = os.getcwd()
    if imgPath is not None and not os.path.isabs(imgPath):
      imgPath = os.path.join(cwd, imgPath)
    confidence=float(confidence)

    script_path = os.path.dirname(os.path.abspath(__file__))

    prototxt = os.path.join(os.path.join(script_path, 'models'), 'deploy.prototxt.txt')
    model = os.path.join(os.path.join(script_path, 'models'), 'res10_300x300_ssd_iter_140000.caffemodel')

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    size = 4096
    size = 2048
    size = 1024
    size = 300
    size = 1024
    size = 2048
    size = 300

    image = cv2.imread(imgPath)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (size, size)), 1.0, (size, size), (103.93, 116.77, 123.68))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
      curConfidence = detections[0, 0, i, 2]

      if curConfidence > confidence:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(curConfidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.95, (0, 0, 255), 2)

    W = float((screeninfo.get_monitors()[0]).width)*0.5
    height, width, depth = image.shape
    imgScale = W/width
    newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
    newimg = cv2.resize(image,(int(newX),int(newY)))
    #cv2.imwrite("resizeimg.jpg",newimg)


    # show the output image
    cv2.imshow("Output", newimg)
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

