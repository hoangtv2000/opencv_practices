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
+ Blob detector regconizes each blob corresponds to each character in a word (in the most cases).
+ Present result to a dictionary search and choose one of classifier choices for each blob in a word.
+ Each word that is satisfactory is passed to an adaptive classifier as training data.
+ Cut poorly recognized characters (called fragments), in order to mproves the classifier confidence. 
+ A best-first search of the resulting segmentation graph puts fragments back together and recombinate together. Then present result to dictionary.
+ The output of BFS is the best overall distance-based rating. 

The rating according to whether the word was in a dictionary and/or had a sensible arrangement of punctuation around it. For the English version, most of these punctuation rules were hard-coded. 


The modern version of Tesseract adds a new LSTM model. The input image is processed in character-bounding boxes line by line feeding into the LSTM model and giving output.


## Part 2. Object detection with SSD and NMS
## Part 3. Simple Face recognition and Attendance
## Part 4. Augmented Reality
