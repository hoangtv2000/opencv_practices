import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime as dt
from PIL import ImageGrab


# Encode the face
def img_encoding(img_list):
    encode_list = []
    for img in img_list:
        encode_img = face_recognition.face_encodings(img)[0]
        encode_list.append(encode_img)
        return encode_list

# read the name list and append the new face to file
def mark_attendance(name, name_list):
    if name not in name_list:
        # Save the time when stranger is in frame
        now = dt.now()
        dt_string = now.strftime('%H:%M:%S')
        appearance = f'\n{name},{dt_string}'
        return appearance
    return None

# For capturing screen
def capture_screen(bbox = None):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_BGR2RGB)
    return capScr


def main():
    path = 'face recog & attendance system\\image folder'

    name_img_list = []
    img_list = []
    name_list = []

    name_img_list = os.listdir(path)
    cnt = len(name_img_list)

    for name in name_img_list:
        img_path = f'{path}\\{name}'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
        name_list.append(os.path.splitext(name)[0])

        cnt -= 1
        if cnt == 0:
            break

    f = open('face recog & attendance system\\names.txt', 'r+')
    data_list = f.readlines()
    for line in data_list:
        name_n_time = line.split(',')
        name_list.append(name_n_time[0])


    # Encode all stored - known faces
    encode_list_known = img_encoding(img_list)

    #--------------------------------------------------------------------------#
    cap = cv2.VideoCapture(0)
    cap.set(3, 960)
    cap.set(4, 480)

    zoom = 1
    scale = 1.0/zoom

    while True:
        _, img = cap.read()
        # img = captureScreen()
        img = cv2.resize(img, (0, 0), None, scale, scale)
        # print(img.shape)
        timer = cv2.getTickCount()

        try:
            # Find locations and encodes of <many> faces
            img_loc = face_recognition.face_locations(img)
            img_encode = face_recognition.face_encodings(img)

        except IndexError:
            continue

        # Get individual
        for face_loc, face_enc in zip(img_loc, img_encode):
            matches = face_recognition.compare_faces(encode_list_known, face_enc)
            dis = face_recognition.face_distance(encode_list_known, face_enc)

            # Face has minimum distance score is accepted
            accept_match = np.argmin(dis)

            # Plot bbox
            y1,x2,y2,x1 = face_loc
            y1,x2,y2,x1 = y1*zoom, x2*zoom, y2*zoom, x1*zoom

            cv2.rectangle(img, (x1,y1), (x2,y2), (255, 127, 0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (255, 127, 0), cv2.FILLED)

            name = None
            if matches[accept_match]:
                name = name_list[accept_match].upper()
            else:
                name = 'Unknown'
                # Save the face of stranger
                appearance = mark_attendance(name, name_list)

            name_list.append(name)

            try:
                f.writelines(appearance)
            except TypeError:
                pass

            # Name
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (129, 245, 251), 2)
            # FPS
            fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
            info = 'fps:{}'.format(str(int(fps)))
            cv2.putText(img, info, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)

            cv2.imshow('Webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break


    cv2.destroyAllWindows()
    f.close()


if __name__ == "__main__":
    main()
