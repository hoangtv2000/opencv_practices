# ==========import the necessary packages============
import numpy as np
import cv2

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


def reorder_points(points):
    """Reoder points of quadrilateral.
    """
    points = points.reshape((4,2))
    ordered_points = np.zeros((4,1,2), dtype=np.float32)

    sum = points.sum(1)
    diff = np.diff(points, axis=1)

    ordered_points[0] = points[np.argmin(sum)]  # [0, 0]
    ordered_points[1] = points[np.argmin(diff)] # [w, 0]
    ordered_points[2] = points[np.argmax(diff)] # [0, h]
    ordered_points[3] = points[np.argmax(sum)]  # [w, h]

    return ordered_points


def processing(path, transformation = False, out_sz = 360):
    """From Image to Bird's-eye-view Grid of Sudoku.
    """
    img = cv2.imread(path)
    width, height, _ = img.shape

    if transformation == True:
        # 1. PREPROCESSING
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # Elipse kernel
        elipse_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        # For getting cleared image
        closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, elipse_ker)
        img_clear = np.float32(img_gray)/(closed)

        img_processed = np.uint8(cv2.normalize(img_clear, img_clear, 0, 255, cv2.NORM_MINMAX))
        img_processed_coled = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)


        # 2. Finding Sudoku Square and Creating Mask Image
        img_thresh = cv2.adaptiveThreshold(img_processed, 255, 0, 1, 19, 2)
        cnts, hier = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        biggest_cnts = np.array([]) #blank array
        max_area = 0.0

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 500:
                #Calculate length of arc
                peri = cv2.arcLength(cnt, True)
                #Determine peaks of polygons
                approx = cv2.approxPolyDP(cnt, 0.1*peri, True)

                if area >= max_area and len(approx) == 4:
                    biggest_cnts = approx
                    max_area = area

        biggest_cnts = reorder_points(biggest_cnts)
        biggest_cnts = np.float32(biggest_cnts)

        img_point = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(biggest_cnts, img_point)

        img_warped = cv2.warpPerspective(img_processed, matrix, (width, height))
        img_warped_coled = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR)


        #3. Finding Vertical lines
        ver_ker = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 12))
        # Applying Sobel operator
        ver_grad = cv2.Sobel(img_warped, cv2.CV_16S, 1, 0)
        ver_grad = cv2.convertScaleAbs(ver_grad)

        _, thresh = cv2.threshold(ver_grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # For filling the blank space in size the digit and columns
        ver_closed = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, ver_ker, iterations = 1)
        # Getting contours and determining contours of vertical lines
        cnts, hier = cv2.findContours(ver_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find major lines by contours
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if h/w > 5:
                cv2.drawContours(ver_closed, [cnt], 0, 255, -1)
            else:
                cv2.drawContours(ver_closed, [cnt], 0, 0, -1)

        ver_closed = cv2.morphologyEx(ver_closed, cv2.MORPH_CLOSE, None, iterations = 2)


        #4. Finding Horizontal lines
        hor_ker = cv2.getStructuringElement(cv2.MORPH_RECT,(12, 4))
        # Applying Sobel operator
        hor_grad = cv2.Sobel(img_warped, cv2.CV_16S, 0, 1)
        hor_grad = cv2.convertScaleAbs(hor_grad)

        _, thresh = cv2.threshold(hor_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # For filling the blank space in size the digit and columns
        hor_closed = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, hor_ker, iterations = 1)
        # Getting contours and determining contours of vertical lines
        cnts, hier = cv2.findContours(hor_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find major lines by contours
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w/h > 5:
                cv2.drawContours(hor_closed, [cnt], 0, 255, -1)
            else:
                cv2.drawContours(hor_closed, [cnt], 0, 0, -1)

        hor_closed = cv2.morphologyEx(hor_closed, cv2.MORPH_CLOSE, None, iterations = 2)


        #5. Finding Grid points
        closed = cv2.bitwise_and(hor_closed, ver_closed)


        #6. Counting grid points
        cnts, hier = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        grid_points = np.array([])
        for cnt in cnts:
            mom = cv2.moments(cnt)
            x,y = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            # DRAW POINTS
            # cv2.circle(img_warped_coled, (x,y), 4, (0,255,0), -1)
            grid_points = np.append(grid_points, [x, y])

        grid_points = grid_points.reshape((100, 2))

        idx_sorted_y = np.argsort(grid_points[:,1])
        sorted_grid_points_y = grid_points[idx_sorted_y]

        sorted_grid_pts = np.array([])

        for i in range(10):
            sorted_x_per_y = sorted_grid_points_y[i*10:(i+1)*10][np.argsort(sorted_grid_points_y[i*10:(i+1)*10, 0])]
            sorted_grid_pts = np.append(sorted_grid_pts, sorted_x_per_y)

        sorted_grid_pts = np.vstack(sorted_grid_pts)
        reshaped_pts = sorted_grid_pts.reshape((10, 10, 2))


        #7. transformation to bird's-eye view of grid
        grid_sz = 150
        res = np.zeros((grid_sz*9, grid_sz*9, 3), np.uint8)

        for x, y in enumerate(sorted_grid_pts):
            row_x = int(x/10)
            col_x = int(x%10)
            if row_x != 9 and col_x != 9:
                try:
                    src = reshaped_pts[row_x:row_x+2, col_x:col_x+2, :].astype(np.float32).reshape((4,2))
                    dst = np.array([[col_x*grid_sz, row_x*grid_sz], [(col_x+1)*(grid_sz-1), row_x*grid_sz],\
                     [col_x*grid_sz, (row_x+1)*(grid_sz-1)], [(col_x+1)*(grid_sz-1), (row_x+1)*(grid_sz-1)]], np.float32)
                    # print('src:', src, '\n', 'dst:', dst)
                    matrix = cv2.getPerspectiveTransform(src, dst)

                    warped = cv2.warpPerspective(img_warped_coled, matrix, (grid_sz*10, grid_sz*10))
                    res[(row_x*grid_sz):(row_x+1)*(grid_sz-1), (col_x*grid_sz):(col_x+1)*(grid_sz-1)] =\
                     warped[(row_x*grid_sz):(row_x+1)*(grid_sz-1), (col_x*grid_sz):(col_x+1)*(grid_sz-1)].copy()

                except ValueError:
                    pass

        #8. Scale result (optional)
        res = cv2.resize(res, (out_sz, out_sz))

        return res, biggest_cnts, img_warped_coled, closed
        # return biggest_cnts, img_warped_coled, closed, hor_closed, ver_closed
    return img, width, height


def split_boxes(img):
    rows = np.vsplit(img, 9)
    boxes=[]
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes
