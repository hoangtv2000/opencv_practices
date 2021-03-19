import cv2
import numpy as np

#Stack images horizontally
def stack_imgs(scale, image_list, labels = []):
    """Return stacked image for all images image_list.
    """
    rows = len(image_list)
    cols = len(image_list[0])
    rows_available = isinstance(image_list[0],list)
    wid = image_list[0][0].shape[1]
    hei = image_list[0][0].shape[0]
    #have columns
    if rows_available:
        for x in range(rows):
            for y in range(cols):
                if image_list[x][y].shape[:2] == image_list[0][0].shape[:2]:
                    image_list[x][y] = cv2.resize(image_list[x][y],(0,0),None,scale,scale)
                else:
                    image_list[x][y] = cv2.resize(image_list[x][y],(image_list[0][0].shape[1],image_list[0][0].shape[0]),None,scale,scale)
                if len(image_list[x][y].shape) == 2:
                    image_list[x][y]= cv2.cvtColor(image_list[x][y], cv2.COLOR_GRAY2BGR)
        blank_img = np.zeros((hei,wid,3), np.uint8)
        hor = [blank_img]*rows
        hor_con = [blank_img]*rows
        for x in range(rows):
            hor[x] = np.hstack(image_list[x])
        ver = np.vstack(hor)
    #no column
    else:
        for x in range(rows):
            if image_list[x].shape[:2] == image_list[0].shape[:2]:
                image_list[x] = cv2.resize(image_list[x], (0, 0), None, scale, scale)
            else:
                image_list[x] = cv2.resize(image_list[x], (image_list[0].shape[1], image_list[0].shape[0]), None,scale, scale)
            if len(image_list[x].shape) == 2:
                image_list[x] = cv2.cvtColor(image_list[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(image_list)
        ver = hor

    if len(labels) != 0:
        each_img_w = int(ver.shape[1]/cols)
        each_img_h = int(ver.shape[0]/rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c*each_img_w, each_img_h*d),(c*each_img_w + len(labels[d])*13+27, 30+each_img_h*d),(255,255,255), cv2.FILLED)
                cv2.putText(ver, labels[d], (each_img_w*c+10, each_img_h*d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,255), 2)
    return ver


def img2canny(image):
    """Edge detect image by Canny edge detector.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)
    return img_canny


def rect_cnts(cnts):
    """Return list of quadrilateral from big area to small.
    """
    list_rect = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 100:
            #Calculate length of arc
            perimeter = cv2.arcLength(cnt, True)
            #Determine peaks of polygons
            approx = cv2.approxPolyDP(cnt, 0.1*perimeter, True)
            if len(approx) == 4:
                list_rect.append(cnt)

    list_rect =sorted(list_rect, key = cv2.contourArea, reverse = True)
    return list_rect


def rect_corner_points(rec_cnts):
    """Return corner points of polygons.
    """
    perimeter = cv2.arcLength(rec_cnts, True)
    approx = cv2.approxPolyDP(rec_cnts, 0.1*perimeter, True)
    return approx


def reorder_points(points):
    """Reoder points of quadrilateral.
    """
    points = points.reshape((4,2))
    ordered_points = np.zeros((4, 1, 2,), dtype=np.int32)

    sum = points.sum(1)
    diff = np.diff(points, axis=1)

    ordered_points[0] = points[np.argmin(sum)]  # [0, 0]
    ordered_points[1] = points[np.argmin(diff)] # [w, 0]
    ordered_points[2] = points[np.argmax(diff)] # [0, h]
    ordered_points[3] = points[np.argmax(sum)]  # [w, h]

    return ordered_points


def split_boxes(img, rows, cols):
    """Split the answer sheet to boxes.
    """
    rows = np.vsplit(img, rows)
    answer_boxes = []
    for row in rows:
        boxes = np.hsplit(row, cols)
        for box in boxes:
            answer_boxes.append(box)
    return answer_boxes


def display_answers(img_warpped_ans, grading, answers, right_answers, questions, choices):
    """Display detected answers and right answer_boxes.
    """
    # For seperating location of answers
    sec_W = int(img_warpped_ans.shape[1]/questions)
    sec_H = int(img_warpped_ans.shape[0]/choices)

    for que in range(questions):
        ans = answers[que]
        # Displaying centers of circle
        cX = ans*sec_W + sec_W//2
        cY = que*sec_H + sec_H//2

        true_color = (0,255,0)

        if grading[que] == 1:
            cv2.circle(img_warpped_ans, (cX, cY), 80, true_color, thickness = 10)

        if grading[que] == 0:
            false_color = (0,0,255)
            cv2.circle(img_warpped_ans, (cX, cY), 80, false_color, thickness = 10)

        # Right answer
        right_ans = right_answers[que]
        right_cX = right_ans*sec_W + sec_W//2
        right_cY = que*sec_H + sec_H//2
        cv2.circle(img_warpped_ans, (right_cX, right_cY), 80, true_color, thickness = 10)


def draw_grid(img_warpped_ans, questions, choices):
    """Draw grids for seperating answer boxes.
    """
    sec_W = int(img_warpped_ans.shape[1]/questions)
    sec_H = int(img_warpped_ans.shape[0]/choices)

    for i in range(questions*choices):
        pt1 = (0, sec_H*i)
        pt2 = (img_warpped_ans.shape[1], sec_H*i)
        pt3 = (sec_W * i, 0)
        pt4 = (sec_W*i, img_warpped_ans.shape[0])
        cv2.line(img_warpped_ans, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img_warpped_ans, pt3, pt4, (255, 255, 0), 2)
