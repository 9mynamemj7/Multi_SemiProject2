import cv2
def handle_img(
                        img_path,
                        x = 128,
                        y= 128,
                        blur = True,
                        edge = True):

    img = cv2.imread(img_path)
    resized = cv2.resize(img, (x, y))
    blured = resized.copy()
    if blur == True: 
        blured = cv2.GaussianBlur(resized, ksize= (5, 5), sigmaX = 0)
    edged = resized.copy()
    if edge == True:
        edged = cv2.Canny(blured, 10, 200)
    return edged