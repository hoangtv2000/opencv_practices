import face_recognition
import cv2
import numpy as np

og_image = face_recognition.load_image_file('face recog & attendance system\\me.jpg')
og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)

og_loc = face_recognition.face_locations(og_image)[0]
encode_og = face_recognition.face_encodings(og_image)[0]

# cv2.rectangle(og_image, (og_loc[3], og_loc[0]),(og_loc[1], og_loc[2]), (0, 255, 255) , 2)

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 360)

while True:
    try:
        timer = cv2.getTickCount()
        _, image = cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        loc = face_recognition.face_locations(image)[0]
        encode = face_recognition.face_encodings(image)[0]

        cv2.rectangle(image, (loc[3], loc[0]),(loc[1], loc[2]), (0, 255, 255) , 2)

        results = face_recognition.compare_faces([encode_og], encode)
        faceDis = face_recognition.face_distance([encode_og], encode)

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
