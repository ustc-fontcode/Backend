# Chinese Font-Code Backend
## introduction
a project for creating encoder and decoder based on Chinese font

## requirement
- pytorch
- pillow
- flask

## quick start
```bash
python app.py
```
visit [localapp](http://localhost:5000/predict) to have a check


## wordcut usage
```python
from wordcut import wordcut,pretreat

# example
font_list = [config.FONT_NAME_Micro, config.FONT_NAME_HuaWen, config.FONT_NAME_Sun, config.FONT_NAME_Fangzheng]
for font in font_list:
    wordcut.cut_word_with_size_and_border.count = 0
    path = "data/{}/".format(font)
    files = os.listdir(path)
    for f in files:
        print(f)
        img = Image.open(path + f)
        img = pretreat.crop_image(img)
        # img.show()
        chars_list = cut_word.cut_word_with_size_and_border(img, font, wordcut.cut_word_with_size_and_border.count)
```