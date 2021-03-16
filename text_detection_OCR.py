import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\\Apps\\tesseract\\tesseract.exe'

# Detecting characters
def character_detector():
    image = cv2.imread('1.png')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_image, w_image,_  = image.shape
    chars_bboxes = pytesseract.image_to_boxes(image)

    for bbox in chars_bboxes.splitlines():
        bbox = bbox.split(' ')
        print(bbox)
        # bbox[0] = info, bbox[1:4] = bbox location
        x, y, width, height = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])

        #plot bboxes and info
        cv2.rectangle(image, (x, h_image - y), (width, h_image - height), (50, 50, 255), 1)
        # cv2.putText(image, bbox[0], (x, h_image - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,55), 2)

    cv2.imshow('Image and characters', image)
    cv2.waitKey(0)

# Detecting words
def word_detector():
    image = cv2.imread('1.png')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_image, w_image,_  = image.shape
    words_bboxes = pytesseract.image_to_data(image)

    for a, b in enumerate(words_bboxes.splitlines()):
        if(a != 0):
            b = b.split( )
            if(len(b) == 12):
                x, y, width, height = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.rectangle(image, (x, y), (x + width, y + height), (50, 255, 50), 1)
                cv2.putText(image, b[-1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50),1) # Last ellement is text

    cv2.imshow('Image and words', image)
    cv2.waitKey(0)

# Detecting words on webcam or on-screen
def realtime_detector():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    def captureScreen(bbox = None):
        capScr = np.array(ImageGrab.grab(bbox))
        capScr = cv2.cvtColor(capScr, cv2.COLOR_BGR2RGB)
        return capScr

    while True:
        timer = cv2.getTickCount()
        _, image = cap.read()
        # image = captureScreen()
        h_image, w_image, _ = image.shape
        words_bboxes = pytesseract.image_to_data(image)

        for a, b in enumerate(words_bboxes.splitlines()):
            if(a != 0):
                b = b.split( )
                if(len(b) == 12):
                    x, y, width, height = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.rectangle(image, (x, y), (x + width, y + height), (50, 255, 50), 1)
                    cv2.putText(image, b[-1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 1)
        # check fps
        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        cv2.putText(image, str(int(fps)), (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 1)

        cv2.imshow('Realtime camera OCR', image)

        if cv2.waitKey(1) == ord('x'):
            break

    cv2.destroyAllWindows()

#TODO: call func you want to use
