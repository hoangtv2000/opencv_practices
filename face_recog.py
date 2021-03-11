import face_recognition
import cv2
import numpy as np

#Load image and covert to BGR -> RGB
og_image = face_recognition.load_image_file('hoangnl2.jpg')
og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)

aff_image = face_recognition.load_image_file('hoangnl1.jpg')
aff_image = cv2.cvtColor(aff_image, cv2.COLOR_BGR2RGB)
#------------------------------------------------------------

# Find face location and encode by HOG feature descriptor
og_loc = face_recognition.face_locations(og_image)[0]
encode_og = face_recognition.face_encodings(og_image)[0]
cv2.rectangle(og_image, (og_loc[3], og_loc[0]),(og_loc[1], og_loc[2]), (0, 255, 255) , 2)

aff_loc = face_recognition.face_locations(aff_image)[0]
encode_aff = face_recognition.face_encodings(aff_image)[0]
cv2.rectangle(aff_image, (aff_loc[3], aff_loc[0]), (aff_loc[1], aff_loc[2]), (0, 255, 255) , 2)
#------------------------------------------------------------

# Compare and calculate face distance
results = face_recognition.compare_faces([encode_og], encode_aff)
faceDis = face_recognition.face_distance([encode_og], encode_aff)

cv2.putText(aff_image, f'{results} {round(faceDis[0],2)} ',(50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
#------------------------------------------------------------

# Plot!
height, width = 1200, 600
cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('original image', width, height)
cv2.imshow('original image', og_image)

cv2.imshow('affine transformed image', aff_image)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
