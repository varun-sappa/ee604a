import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    def detect_lines(image_path):
        src = cv2.imread(image_path)
        pre_dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(pre_dst, (7, 7), 0)
        _, binary = cv2.threshold(blurred1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dst = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(dst, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=20)
        
        return lines

    src = cv2.imread(image_path)

    pre_dst1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred11 = cv2.GaussianBlur(pre_dst1, (7, 7), 0)
    _, binary1 = cv2.threshold(blurred11, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst11 = cv2.Canny(binary1, 50, 200, apertureSize=3)
    lines1111 = cv2.HoughLinesP(dst11, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=20)

    lines=detect_lines(image_path)

    if lines is not None:
        angles = []
        temp=src.copy()

        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
            cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines on the rotated image


        average_angle = np.degrees(np.median(angles))
        if average_angle < 0:
            final_angle = average_angle + 180
        else:
            final_angle = average_angle

        height, width = src.shape[:2]
        image_center = (width/2, height/2)
        rotation_arr = cv2.getRotationMatrix2D(image_center, final_angle, 1)
        
        abs_cos = abs(rotation_arr[0,0])
        abs_sin = abs(rotation_arr[0,1])
        
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        
        rotation_arr[0, 2] += bound_w/2 - image_center[0]
        rotation_arr[1, 2] += bound_h/2 - image_center[1]
        
        white_background = np.ones((bound_h, bound_w, 3), dtype=np.uint8) * 255 
        final = cv2.warpAffine(src, rotation_arr, (bound_w, bound_h), borderValue=(255, 255, 255))

    def flip():
        src1=final.copy()
        gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        dst1 = cv2.Canny(gray, 50, 150)
        lines1 = cv2.HoughLinesP(dst1, 1, np.pi / 180, threshold=35, minLineLength=25, maxLineGap=1)

        if lines1 is not None:
            lines1 = sorted(lines1, key=lambda line: line[0][1])
            lowest_line = lines1[-1][-1]

            x1, y1, x2, y2 = lowest_line
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            side=9
            roi = gray[center_y-side:center_y+side, center_x-side:center_x+side]
            threshold_value = 128  # You can adjust this threshold value as needed
            roi_1 = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)[1]

            pixels_above = np.sum(roi[0:side-2] < 128)
            pixels_below = np.sum(roi[side+2:2*side] <128)


            cv2.rectangle(src1, (center_x-side, center_y-side), (center_x+side, center_y+side), (0, 0, 255), 1)

            if(pixels_above>pixels_below):
                return 1
            else:
                return 0
            

    if(flip()):
        image = cv2.rotate(final, cv2.ROTATE_180)
    else:
        image = final.copy()
    ############################

    return image




