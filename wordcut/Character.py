from PIL import Image
import cv2
import numpy as np


class ChineseCharacter:

    def __init__(self, font: str, value: str, image: Image.Image, source: Image.Image =None, location: list =None):
        self.font = font
        self.value = value if value != '' and value != '.' and value != '\\' and value != '\/' else 'None'
        print(value)
        self.image = image
        self.source = source
        self.location = location

    def from_tesseract_box(self, statement: str, img: Image.Image, font: str):
        font = font.split(".")[0]  # 把后缀去掉

        data = statement.split(" ")

        value = data[0]
        image = img.crop((self.x1, self.y1, self.x2, self.y2))

        w, h = img.size
        source = img
        location = [int(data[1]), h - int(data[4]), int(data[3]), h - int(data[2])]
        # self.x1, self.y2, self.x2, self.y1 =  h - int(data[2]), int(data[3]), h - int(data[4])
        return ChineseCharacter(font, value, image, source=source, location=location)

    # def from_cut_size(self, source: Image.Image, )

    def save(self):
        self.image.save("result/{}/{}.png".format(self.font, self.value))

    def show(self):
        img = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)
        cv2.waitKey()
