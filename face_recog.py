import face_recognition
import cv2
import numpy as np

#Load image and covert to BGR -> RGB
def load_n_convert(path):
    img = face_recognition.load_image_file(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Find face location and encode by HOG feature descriptor
def face_loc_n_encode(image):
    loc = face_recognition.face_locations(image)[0]
    encode = face_recognition.face_encodings(image)[0]

    return loc, encode
#-------------------------------------------------------------------------------

def main():
    og_path = 'face recog & attendance system\hoangnl2.jpg'
    aff_path = 'face recog & attendance system\hoangnl1.jpg'

    # Load 2 images
    og_image = load_n_convert(og_path)
    aff_image = load_n_convert(aff_path)

    # Find location and encode image
    og_loc, og_encode = face_loc_n_encode(og_image)
    aff_loc, aff_encode = face_loc_n_encode(aff_image)

    # Compare and calculate face distance
    results = face_recognition.compare_faces([og_encode], aff_encode)
    faceDis = face_recognition.face_distance([og_encode], aff_encode)

    # Plot!
    height, width = 1200, 600

    # OG image
    cv2.rectangle(og_image, (og_loc[3], og_loc[0]), (og_loc[1], og_loc[2]), (0, 255, 255) , 2)
    cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original image', width, height)
    cv2.imshow('original image', og_image)

    # Transformed image
    cv2.rectangle(aff_image, (aff_loc[3], aff_loc[0]), (aff_loc[1], aff_loc[2]), (0, 255, 255) , 2)
    cv2.putText(aff_image, f'{results} {round(faceDis[0],2)} ',(50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    cv2.namedWindow('affine transformed image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('affine transformed image', width, height)
    cv2.imshow('affine transformed image', aff_image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
