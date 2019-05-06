import config
import numpy as np
from PIL import Image
import cv2


def pretreat(img: Image.Image)->Image.Image:
    img = crop_image(img)
    img_data = np.array(img)
    img_data = np.mean(img_data, -1)  # 将rgb转为灰度图
    # 二值化
    img_data[img_data > 150] = 255

    img_data[img_data <= 100] = 0
    img = Image.fromarray(img_data)
    return img

'''
def crop_image(image):
    """Crop largest box from and return as pillow image.
    """
    image = image.resize((image.size[0] // 2, image.size[1] // 2))  # Shrink to make boarder clear
    image = np.asarray(image)
    # Generate edge image
    edge_img = cv2.GaussianBlur(image, (3,3), 0)
    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(edge_img, 50, 150, apertureSize=3)
    # Find contour
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea)[-1]
    epsilon = 0.1*cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    def four_point_transform(image, pts):
        def order_points(pts):
            rect = np.zeros((4, 2), dtype = "float32")

            s = pts.sum(axis = 1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis = 1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            return rect

        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]],
            dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    paper = four_point_transform(image, contour.reshape(4, 2))
    return Image.fromarray(paper).resize(config.IMAGE_SIZE)
'''

def crop_image(image):
    """Crop document from image.
    
        Takes a PIL.image and return a PIL.image. Not resized.
    """
    def rectify(h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

    # resize image so it can be processed
    # choose optimal dimensions such that important content is not lost
    image = np.asarray(image)
    image = cv2.resize(image, (1500, 880))

    # creating copy of original image
    orig = image.copy()

    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #blurred = cv2.medianBlur(gray, 5)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)
    orig_edged = edged.copy()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #x,y,w,h = cv2.boundingRect(contours[0])
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break

    # mapping target points to 800x800 quadrilateral
    approx = rectify(target)
    pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

    M = cv2.getPerspectiveTransform(approx,pts2)
    dst = cv2.warpPerspective(orig,M,(800,800))

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    
    return Image.fromarray(dst).resize(config.IMAGE_SIZE)


if __name__ == "__main__":
    img = Image.open("data/1.jpg")
    img = crop_image(img)
    img.show()
