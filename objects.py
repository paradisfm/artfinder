import cv2

img = cv2.imread(r"C:\Users\hadar\misc_python\art_finder.py\SAMPLE.png")

def contour_shapes(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [contour], 0, (0,0,255), 5)

        M = cv2.moments(contour)
        if M['m00'] != 0.0: 
            x = int(M['m10']/M['m00']) 
            y = int(M['m01']/M['m00']) 

    cv2.imshow('shapes', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()