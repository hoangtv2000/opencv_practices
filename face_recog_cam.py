import face_recognition
import cv2
import numpy as np
import face_recog

path = 'face recog & attendance system\\me.jpg'

og_img = face_recog.load_n_convert(path)
_, og_encode = face_recog.face_loc_n_encode(og_img)


cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 360)

while True:
    try:
        timer = cv2.getTickCount()
        _, image = cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        loc, encode = face_recog.face_loc_n_encode(image)

        cv2.rectangle(image, (loc[3], loc[0]),(loc[1], loc[2]), (0, 255, 255) , 2)

        results = face_recognition.compare_faces([og_encode], encode)
        faceDis = face_recognition.face_distance([og_encode], encode)

        cv2.putText(image, f'{results} {round(faceDis[0],2)} ',(50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255 ,0), 2)

        # FPS
        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        info = 'fps:{}'.format(str(int(fps)))
        cv2.putText(image, info, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)

        cv2.imshow("Output", image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    # continue the loop if face is not detected in particular frame
    except IndexError:
        continue


cv2.destroyAllWindows()
