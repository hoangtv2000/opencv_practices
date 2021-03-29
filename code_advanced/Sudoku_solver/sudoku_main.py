import os
import cv2
import numpy as np
from sudoku_processing import *
from sudoku_solver import *

path = 'sudoku_solver\\165935292_157247906178377_6453587673404990236_n.jpg'
model_path = 'sudoku_solver\\digit_classifier.h5'

img, width, height = processing(path, transformation=False)

model = call_model(model_path)

img_grid, biggest_cnts, img_warped_coled, closed = processing(path, transformation=True)

digits = digit_extract(img_grid, model)
# grid_display  = display_digit(img_grid, digits)

# Images for plotting
img_display_digit = img_grid.copy()
img_inv_wrap_cold = img.copy()


digits = np.array(digits)
# Split digits arr to 9 cols
grid = np.array_split(digits, 9)
# Check solved digit to get index of TO_DO digit
solved_idx = np.where(digits > 0, 0, 1)

# Solve Sudoku
sudoku_solve(grid)
# Display solved grid by output
display_grid(grid)

flat_grid = []

for row in grid:
    for digit in row:
        flat_grid.append(digit)

# Choose TO_DO digit to draw
solved_digits = flat_grid*solved_idx

display_digit(img_display_digit, solved_digits, color = (255, 0, 255))

img_display_digit = cv2.resize(img_display_digit, (width, height))


"""OVERLAY"""

# Compute invert matrix to unwarp image
grid_cnts = np.float32(biggest_cnts)
img_cnts =  np.float32([[0, 0],[width, 0], [0, height],[width, height]])

inv_matrix = cv2.getPerspectiveTransform(img_cnts, grid_cnts)

img_inv_wrap_cold = cv2.warpPerspective(img_display_digit, inv_matrix, (height, width))
# Add background
inv_perspective = cv2.addWeighted(img_inv_wrap_cold, 1, img, 0.4, 1)


"""RESULT"""

stacked_imgs = [[img, img_warped_coled, closed],[img_grid, img_display_digit, inv_perspective]]

stacked_imgs = stack_imgs(0.3, stacked_imgs)

cv2.imwrite('resasd.jpg', stacked_imgs)

cv2.imshow('Test', stacked_imgs)
cv2.waitKey(0)
