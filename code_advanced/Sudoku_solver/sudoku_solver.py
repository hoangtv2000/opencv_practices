import cv2
import numpy as np
from sudoku_processing import *
from tensorflow.keras.models import load_model


"""PRED AND DISPLAY DIGIT"""

def call_model(model_path):
    model = load_model(model_path)
    return model


def digit_extract(img, model):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Split the grid to cells of digit
    boxes = split_boxes(img_gray)

    # Extract digits by pretrained deep model
    digits = []
    for box in boxes:
        # Cut redunant and resize
        box = box[4:-4, 4:-4]
        box = cv2.resize(box, (28, 28))
        # Scale pixel
        box = box / 255
        box = box.reshape(1, 28, 28, 1)
        # Predict digit
        pred_prob = model.predict(box) # return probability
        class_ = model.predict_classes(box) # return class
        prob_max = np.amax(pred_prob) # max prob

        if prob_max > 0.8:
            digits.append(class_[0])
        else:
            digits.append(0)

    return digits


def display_digit(img, digits, color = (255, 0, 255)):
    H = img.shape[0]/9
    W = img.shape[1]/9
    for row in range(9):
        for col in range(9):
            if digits[row*9 + col] != 0:
                 cv2.putText(img, str(digits[row*9 + col]),
                               (int(col*W + W/2-10), int((row+0.8)*H)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img


def display_grid(grid):
    for i in range(len(grid)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(grid[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(grid[i][j])
            else:
                print(str(grid[i][j]) + " ", end="")


"""SUDOKU SOLVER"""


def find_empty_location(grid, position):
    """Find empty position.
    """
    for row in range(9):
        for col in range(9):
            if(grid[row][col] == 0):
                position[0]= row
                position[1]= col
                return True
    return False


def used_in_row(grid, row, digit):
    """Return True if digit used in row.
    """
    for i in range(9):
        if(grid[row][i] == digit):
            return True
    return False


def used_in_col(grid, col, digit):
    """Return True if digit used in col.
    """
    for i in range(9):
        if(grid[i][col] == digit):
            return True
    return False


def used_in_box(grid, row, col, digit):
    """Return True if digit used in 3x3 box.
    """
    for i in range(3):
        for j in range(3):
            if(grid[i + row][j + col] == digit):
                return True
    return False


def check_location_is_safe(grid, row, col, digit):
    """Check if digit is not already placed in current row, current column and current 3x3 box.
    """
    return not used_in_row(grid, row, digit) and not used_in_col(grid, col, digit)\
        and not used_in_box(grid, row - row % 3, col - col % 3, digit)


def sudoku_solve(grid):
    """Solve Sudoku main function.
    """
    position =[0, 0]
    if(not find_empty_location(grid, position)):
        return True

    row = position[0]
    col = position[1]

    for digit in range(1, 10):
        if(check_location_is_safe(grid, row, col, digit)):
            # make tentative assignment
            grid[row][col]= digit
            # return, if success,
            if(sudoku_solve(grid)):
                return True
            # failure, unmake & try again
            grid[row][col] = 0

    # this triggers backtracking
    return False
