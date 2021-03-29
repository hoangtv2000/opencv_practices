# Sudoku Solver

We solve captured image of Sudoku by using hand-craft Image processing technique, deep learning and Backtracking algorithm.

**Objective**: Automatically capture the Sudoku grid, detect cells have given digits or blankspaces. 
After that, we solve the Sudoku grid by Backtracking algorithm and unwarp the grid into initial status.

**Input**: An image contains a non or partially convex Sudoku grid.

**Output**: An image with solved Sudoku grid.

[Click here to discover the code](https://github.com/hoangtv2000/opencv_practices/blob/main/code_advanced/Sudoku_solver)


## Process to solve

+ Step 1: Provide input image containing Sudoku puzzle to our system.
+ Step 2: Locate where in the input image the grid is and extract the grid.
+ Step 3: Given the grid, locate each of the individual cells of the Sudoku grid (most standard Sudoku puzzles are a 9×9 grid, so we’ll need to localize each of these cells).
+ Step 4: Determine if a digit exists in the cell, and if so, detect given digits by CNN digit classifier.
+ Step 5: Apply a Backtracking algorithm to solve and validate the puzzle.
+ Step 6: Unwarp the grid into initial status and display the output result to the user.


## Input images

<img src = 'https://github.com/hoangtv2000/opencv_practices/blob/main/images/sudoku-og1.jpg' width = '500' height = '600' wspace="20"/><img src = 'https://github.com/hoangtv2000/opencv_practices/blob/main/images/sudoku-og2.jpg' width = '500' height = '600'/>

