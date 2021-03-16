# OPENCV practices 
## Part 1. Simple Optical character recognition by Tesseract
Tesseract is the OCR (Optical Character Recognition) current top engine, it is developed by Google, with open-source license Apache 2.0. In the first part, we applicate the engine to build a simple OCR program.

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

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/images/part1_res.png" alt="Part1 result">


## Part 2. Object detection with SSD and NMS
Implement Single Shot Detector (SSD) for object detection problem. And apply Non-max suppression techique to remove redundant proposal boxes.

### SSD
**Single shot detector (SSD)** is a NN architecture designed for detection purposes - which means localization (bounding boxes) and classification at once.

**MobileNet** is the compact and efficient **feature extractor**, using Depth-wise Separable convolution layers to build a light weight deep neural net.

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/images/mobileNet-SSD-network-architecture.png" alt="MobileNet SSD Architecture">


### NMS
**Input**: A list of proposal boxes B, corresponding confidence scores S and overlap threshold N.

**Output**: A list of filtered proposals D.

#### NMS process
+ **Step 1.** Create empty D, select the proposal with highest confidence score, remove it from B and add it to the final proposal list D. 
+ **Step 2.** Compare this proposal with all remaining proposals in B — calculate the Intersection over Union (IOU) of this proposal with every other proposal. If the IOU is greater than the threshold N, remove that proposal from B.
+ **Step 3.** Again take the proposal with the highest confidence from the remaining proposals in B and remove it from B and add it to D.
+ **Step 4.** Once again calculate the IOU of this proposal with all the proposals in B and eliminate the boxes which have high IOU than threshold.
+ **Step 5.** This process is repeated until there are no more proposals left in B.

### Result

<img src="https://github.com/hoangtv2000/opencv_practices/blob/main/images/part2_res.png" alt="Part2 result">

## Part 3. Simple Face recognition and Attendance


## Part 4. Augmented Reality
