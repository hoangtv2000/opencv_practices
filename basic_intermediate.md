## OPENCV practices 
# Part 1. Simple Optical character recognition by Tesseract
Tesseract is the OCR (Optical Character Recognition) current top engine, it is developed by Google, with open-source license Apache 2.0. In the first part, we applicate the engine to build a simple OCR program.

[Click here to discover the code](https://github.com/hoangtv2000/opencv_practices/blob/main/code_basic_intermediate/text_detection_OCR.py)

### Mechanism
**Input**: Binary Image.

**Output**: Text information.

#### End-to-end Process: 
+ **Page Layout Analysis**: recognize Component outlines in the Text regions.
+ **Blob finding**: recognize Character outline in the Text regions.
+ **Find text lines and words**: organize character outlines in to words.
+ **Recognize words Pass 1 and Pass 2**: Word recoginition.
+ **Fuzzy-space resolution**.

#### Word recognition process:
+ **Step 1.** Blob detector regconizes each blob corresponds to each character in a word (in the most cases).
+ **Step 2.** Present result to a dictionary search and choose one of classifier choices for each blob in a word.
+ **Step 3.** Each word that is satisfactory is passed to an adaptive classifier as training data.
+ **Step 4.** Cut poorly recognized characters (called fragments), in order to mproves the classifier confidence. 
+ **Step 5.** A best-first search of the resulting segmentation graph puts fragments back together and recombinate together. Then  re-present result to dictionary.
+ **Step 6.** The output of BFS is the best overall distance-based rating. 

The rating according to whether the word was in a dictionary and/or had a sensible arrangement of punctuation around it. For the English version, most of these punctuation rules were hard-coded. 

The modern version of Tesseract adds a new LSTM model. The input image is processed in character-bounding boxes line by line feeding into the LSTM model and giving output.

### Result

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/results/part1_res.png" alt="Part1 result">


# Part 2. Object detection with SSD and NMS
Implement Single Shot Detector (SSD) for object detection problem. And apply Non-max suppression techique to remove redundant proposal boxes.

[Click here to discover the code](https://github.com/hoangtv2000/opencv_practices/blob/main/code_basic_intermediate/obj_dec_SSD_NMS.py)

### SSD
**Single shot detector (SSD)** is a neural net architecture designed for object detection purposes - which means extract high-level features, localization (bounding boxes) and classification while propagation. The model use MobileNet-v3 as backbone, and is pretrained by COCO dataset. In can classify [91 diferrent classes](https://github.com/ankityddv/ObjectDetector-OpenCV/blob/main/coco.names). 

**MobileNet** is the compact and efficient **feature extractor**, using **Depth-wise Separable convolution layers** to build a light weight deep neural net.

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/results/mobileNet-SSD-network-architecture.png" alt="MobileNet SSD Architecture">


### NMS
**Objective**: Ignore bad bounding boxes that significantly overlap each other.

**Input**: A list of proposal boxes B, corresponding confidence scores and overlap threshold N.

**Output**: A list of filtered proposals D.

#### NMS process
+ **Step 1.** Create empty D, select the proposal with highest confidence score, remove it from B and add it to the final proposal list D. 
+ **Step 2.** Compare this proposal with all remaining proposals in B — calculate the Intersection over Union (IOU) of this proposal with every other proposal. If the IOU is greater than the threshold N, remove that proposal from B.
+ **Step 3.** Again take the proposal with the highest confidence from the remaining proposals in B and remove it from B and add it to D.
+ **Step 4.** Once again calculate the IOU of this proposal with all the proposals in B and eliminate the boxes which have high IOU than threshold.
+ **Step 5.** This process is repeated until there are no more proposals left in B.

### Result

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/results/part2_res.png" alt="Part2 result">

# Part 3. Simple Face recognition and Attendance

### Face recoginition problem
Solve face recognition problem by face_recognition library. It uses Histogram of Oriented Gradients (HOG) to localize faces. Also uses pretrained neural network as feature descriptor, which has outputs 128 measurements that are unique to particular face. In order to compare faces, we use traditional linear SVM classifier to find whether the face match by the function **compare_faces**, it returns true or false. In addition to calculate the Euclidian distance of two 128-d descriptors, we use **face_distance** function.

We have two programs for this problem, one for recognition by image, another for recogintion by webcam.

[Click here to discover the first program](https://github.com/hoangtv2000/opencv_practices/blob/main/code_basic_intermediate/face_recog.py)

[Click here to discover the second program](https://github.com/hoangtv2000/opencv_practices/blob/main/code_basic_intermediate/face_recog_cam.py)

**Result by comparing two images**

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/results/part3_res1.png" alt="Part3 result 1">

### Attendance problem
Face recognition by webcam and check wether the faces are in our image folder, then save the history of appearance. 

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/results/part3_res2.png" alt="Part3 result 2">
