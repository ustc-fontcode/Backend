# encoding:utf-8
from wordcut import config
import numpy as np
from PIL import Image
import cv2
import sys
import os

def pretreat(img: Image.Image) -> Image.Image:
    img_data = np.array(img)
    # img_data = np.mean(img_data, -1)  # 将rgb转为灰度图
    # 二值化
    img_data[img_data > 128] = 255

    img_data[img_data <= 128] = 0
    img = Image.fromarray(img_data)
    return img


def custom_blur_demo(image: Image.Image, n: int) -> Image.Image:
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    kernel = np.array([[0, -1, 0], [-1, n, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    # cv2.imshow("custom_blur_demo", dst)
    return Image.fromarray(dst)


def crop_image(image: Image.Image) -> Image.Image:
    """Crop document from image.
        Takes a PIL.image and return a PIL.image. Not resized.
    """

    def rectify(h):
        h = h.reshape((4, 2))
        hnew = np.zeros((4, 2), dtype=np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h, axis=1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

    # resize image so it can be processed
    # choose optimal dimensions such that important content is not lost
    image = np.asarray(image)
    # TODO 不知道能不能删掉
    # image = cv2.resize(image, (1500, 880))


    # creating copy of original image
    orig = image.copy()

    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.medianBlur(gray, 5)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)
    # orig_edged = edged.copy()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # x,y,w,h = cv2.boundingRect(contours[0])
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break

    # mapping target points to 800x800 quadrilateral
    approx = rectify(target)
    # pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    pts2 = np.float32([[0, 0], [config.IMAGE_SIZE[0], 0], [config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]], [0, config.IMAGE_SIZE[1]]])

    M = cv2.getPerspectiveTransform(approx, pts2)
    # dst = cv2.warpPerspective(orig, M, (800, 800))
    dst = cv2.warpPerspective(orig, M, config.IMAGE_SIZE)

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # return Image.fromarray(dst).resize(config.IMAGE_SIZE)
    return Image.fromarray(dst)


if __name__ == "__main__":
    '''
    path = "result/" + sys.argv[1] + ".jpg"
    img = Image.open(path)
    img = custom_blur_demo(img, int(sys.argv[2]))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    '''
    # path = "data/newdata/HuaWenSun/"
    # files = os.listdir(path)
    # for f in files:
    #     print(f)
    #     img = Image.open(path + f)
    #     img = crop_image(img)
    #     img.save("data/HuaWenSun/" + f)

    # path = "data/newdata/MicroSun/"
    # files = os.listdir(path)
    # for f in files:
    #     print(f)
    #     img = Image.open(path + f)
    #     img = crop_image(img)
    #     img.save("data/MicroSun/" + f)

    # path = "data/newdata/error-test/"
    path = sys.argv[1]
    files = os.listdir(path)
    for f in files:
        print(f)
        img = Image.open(path + f)
        img = crop_image(img)
        # img.save("data/error-test/" + f)
        img.save(sys.argv[2]+f)

