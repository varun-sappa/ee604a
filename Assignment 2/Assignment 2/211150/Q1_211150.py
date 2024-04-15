import cv2
import numpy as np

# Usage
def solution(image_path):
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    # Read the image
    image = cv2.imread(image_path)
    
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # plt.imshow(cv2.cvtColor(lab, cv2.COLOR_BGR2RGB))
    # plt.show()

    shifted = cv2.pyrMeanShiftFiltering(lab,28,100)
    # shifted = cv2.pyrMeanShiftFiltering(lab,5,5)


    # Convert the image to grayscale
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    # plt.show()

    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)


    # plt.imshow(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
    # plt.show()

    

    result = np.zeros_like(image)
    
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask = np.zeros_like(thresholded)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        result = cv2.bitwise_and(result, result, mask=mask)
    
    # result[thresholded == 255] = [255, 255, 255]

    return result
