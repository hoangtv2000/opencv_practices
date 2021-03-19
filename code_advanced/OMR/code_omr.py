import numpy as np
import cv2
import Opencv_utils as cv_utils

# CONFIG
rows = 5
cols = 5
white_pix_threshold = 10000
questions = 5
choices = 5

right_answers = [1,2,0,0,4]
#-------------------------------------------------------------------------------

mcq_paper = 'OMR\\MCQPaper.jpg'
img_1 = 'OMR\\1.jpg'
img_2 = 'OMR\\2.jpg'

# mcq_paper = cv2.imread(mcq_paper)
# img_1 = cv2.imread(img_1)
# img_2 = cv2.imread(img_2)


img = cv2.imread(mcq_paper)
height, width, _ = img.shape

img_cnts = img.copy()
img_ans_rect = img.copy()
img_grade_rect = img.copy()
img_final = img.copy()

img_canny = cv_utils.img2canny(img)

# FIND CNTS
cnts, hier = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_cnts, cnts, -1, (20, 20, 120), 10) #Draw canny imgs


# FIND RECTS
rect_cnts = cv_utils.rect_cnts(cnts)
# Biggest rect in img => img_ans_rect
ans_rect = cv_utils.rect_corner_points(rect_cnts[0])
# Second biggest rect => Grade
grade_rect = cv_utils.rect_corner_points(rect_cnts[1])

if ans_rect.size != 0 and grade_rect.size != 0:
    cv2.drawContours(img_ans_rect, ans_rect, -1, (20, 20, 120), 40)
    cv2.drawContours(img_grade_rect, grade_rect, -1, (20, 20, 120), 40)

    ans_rect = cv_utils.reorder_points(ans_rect)
    grade_rect = cv_utils.reorder_points(grade_rect)


    # ANS WRAPPING
    # Rect of ans
    ans_pt = np.float32(ans_rect)
    # Rect of image
    img_pt = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Get homo transform matrix to take direct view of ans area
    ans_matrix = cv2.getPerspectiveTransform(ans_pt, img_pt)
    img_warpped_ans = cv2.warpPerspective(img, ans_matrix, (width, height))


    # GRADE WRAPPING
    # Rect of grade
    grade_pt = np.float32(grade_rect)
    # Rect of image
    img_grade_pt =np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    # Get homo transform matrix to take direct view of ans area
    grade_matrix = cv2.getPerspectiveTransform(grade_pt, img_grade_pt)
    img_warpped_grade = cv2.warpPerspective(img, grade_matrix, (width, height))


    # DETECT blob of answer
    img_warpped_ans_gray = cv2.cvtColor(img_warpped_ans, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(img_warpped_ans_gray, 160, 255, cv2.THRESH_BINARY_INV)[1]
    # Split blob of answer into individual box
    splitted_boxes = cv_utils.split_boxes(img_thresh, rows = rows, cols = cols)

    # Calculate white_pixels for each box
    pix_vals = np.zeros((questions, choices))
    count_cols = 0
    count_rows = 0

    for box in splitted_boxes:
        total_pix = np.count_nonzero(box)
        pix_vals[count_rows][count_cols] = total_pix
        count_cols += 1
        if count_cols == choices:
            count_rows += 1
            count_cols = 0

    # Find answers <0:A, 1:B, 2:C, 3:D, 4:E, -1:no answer>
    answers = []
    for que in range(questions):
        arr_of_choices = pix_vals[que]
        if np.max(arr_of_choices) > white_pix_threshold:
            answer_choice = np.where(arr_of_choices == np.max(arr_of_choices))
            answers.append(answer_choice[0][0])
        else:
            answers.append(-1)

    # Compare answers and calculate score
    grading = [] # For knowing each anwser is right or not.
    for que in range(questions):
        if right_answers[que] == answers[que]:
            grading.append(1)
        else:
            grading.append(0)
    score = (sum(grading)/questions)*10


    # DISPLAYING ANSWER
    # Displaying my answers and correct answers.
    cv_utils.display_answers(img_warpped_ans, grading, answers, right_answers, questions, choices)
    cv_utils.draw_grid(img_warpped_ans, questions, choices)

    # Use inverted transformation to display answers
    raw_displaying_ans = np.zeros_like(img_warpped_ans)
    cv_utils.display_answers(raw_displaying_ans, grading, answers, right_answers, questions, choices)
    # Inverted homo transform matrix recieve a initial view
    inv_ans_matrix = cv2.getPerspectiveTransform(img_pt, ans_pt)
    img_inv_warpped_ans = cv2.warpPerspective(raw_displaying_ans, inv_ans_matrix, (width, height))

    # DISPLAYING SCORE
    raw_displaying_grade = np.zeros_like(img_warpped_grade)
    text_score = str(int(score))
    cv2.putText(raw_displaying_grade, text_score, (70,100) ,cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 4)
    inv_grade_matrix = cv2.getPerspectiveTransform(img_grade_pt, grade_pt)
    img_inv_warpped_grade = cv2.warpPerspective(raw_displaying_grade, inv_grade_matrix, (width, height))

    #
    img_final = cv2.addWeighted(img_final, 1, img_inv_warpped_ans, 10, 0)
    img_final = cv2.addWeighted(img_final, 1, img_inv_warpped_grade, 10, 0)



# SHOW
stack_imgs = cv_utils.stack_imgs(0.35, [img, img_final])
# cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Images', 800, 800)
cv2.imshow('Images', stack_imgs)
cv2.waitKey(0)
