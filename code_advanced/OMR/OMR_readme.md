# Optical Mark Regconition (OMR) for multiple choice scanner and test grader

## What is Optical Mark Recognition (OMR)?

Optical Mark Recognition is the process of automatically analyzing human-marked documents and interpreting their results. 
The most famous, easily recognizable form of OMR are bubble sheet multiple choice tests.

**Objective**: Automatically evaluate the right and wrong answers on the bubble sheet multiple choice exam. And write down score to the cell score.

**Input**: An image contains bubble sheet multiple choice exam and array of right answers.

**Output**: Giving a mark for each right and wrong answer, and write down score to the cell score of an exam.

## Process to solve

+ Step 1: Detect the exam in an image.
+ Step 2: Apply a perspective transform to extract the top-down, birds-eye-view of the exam.
+ Step 3: 

