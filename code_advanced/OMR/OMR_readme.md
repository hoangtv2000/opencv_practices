# Optical Mark Regconition (OMR) for multiple choice scanner and test grader

## What is Optical Mark Recognition (OMR)?

Optical Mark Recognition is the process of automatically analyzing human-marked documents and interpreting their results. 
The most famous, easily recognizable form of OMR are bubble sheet multiple choice tests.

**Objective**: Automatically evaluate the right and wrong answers on the bubble sheet multiple choice exam. And write down score to the cell score.

**Input**: An image contains bubble sheet multiple choice exam and array of right answers.

**Output**: Giving a mark for each right and wrong answer, and write down score to the cell score of an exam.

## Process to solve

+ Step 1: Detect the exam in an image.
+ Step 2: Detect four verticles of answer area and grade area of exam.
+ Step 3: Apply a perspective transform to extract the top-down, birds-eye-view of answer area and grade area.
+ Step 4: Binary thresholding the warpped answer area, detect bubble of answers by seperating into grid of bubble cells.
+ Step 5: White-pixel counting bubble cells, the cell have the most white pixels of the row is the marked answer.
+ Step 6: Lookup the right answer to determine whether marked answer was correct.
+ Step 7: Repeat for all questions and calculate score.
+ Step 8: Write down score to grade area.

## Result

Result with the right answers: [B, C, A, A, E]

<img src = 'https://github.com/hoangtv2000/opencv_practices/blob/main/results/omr_res1.png'>
<img src = 'https://github.com/hoangtv2000/opencv_practices/blob/main/results/omr_res2.png'>
