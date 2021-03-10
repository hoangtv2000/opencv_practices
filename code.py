import cv2
import numpy as np

# Import COCO class name
class_name = []
file_name = 'object detection SSD\\coco.names'

with open(file_name, 'rt') as f:
    class_name = f.read().strip('\n').split('\n')

config_path = 'object detection SSD\\frozen_inference_graph.pb'
model_path = 'object detection SSD\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

model = cv2.dnn_DetectionModel(model_path, config_path)

model.setInputSize(640, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# VideoCapture

confidence_threshold = 0.5
nms_threshold = 0.5

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 320)


while True:
    timer = cv2.getTickCount()

    _, image = cap.read()
    class_ids, confs, bbox = model.detect(image, confThreshold = confidence_threshold)
    bbox = list(bbox)
    #reshape
    confs = list(np.array(confs).reshape(1, -1)[0])
    #set all elements list to map
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confidence_threshold, nms_threshold)
    #print(indices)

    for index in indices:
        index = index[0]
        box = bbox[index]
        x, y, width ,height = box[0],box[1],box[2],box[3]
        # cv2.rectangle(image, (x, y), (width, height), color=(0, 255, 0), thickness=1)
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=1)
        cv2.putText(image, class_name[class_ids[index][0] - 1].upper(),(box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    info = 'fps:{}'.format(str(int(fps)))
    cv2.putText(image, info, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 1)

    cv2.imshow("Output", image)
    if cv2.waitKey(1) == ord('x'):
        break

cv2.destroyAllWindows()
