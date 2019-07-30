````````````# Chinese Font-Code Backend

{'huawen': 0, 'micro': 1}

## introduction

a project for creating encoder and decoder based on Chinese font

## requirement

- pytorch
- pillow
- flask
- opencv-python
- pytesseract
- tesseract

## quick start

```bash
python app.py
```

visit [localapp](http://localhost:5000/predict) to have a check

## wordcut usage

```python
from wordcut import cutword,pretreat

# example
font_list = [config.FONT_NAME_Micro, config.FONT_NAME_HuaWen, config.FONT_NAME_Sun, config.FONT_NAME_Fangzheng]
for font in font_list:
    cutword.cut_word_with_size_and_border.count = 0
    path = "data/{}/".format(font)
    files = os.listdir(path)
    for f in files:
        print(f)
        img = Image.open(path + f)
        img = pretreat.crop_image(img)
        # img.show()
        chars_list = cut_word.cut_word_with_size_and_border(img, font, cutoword.cut_word_with_size_and_border.count)
```

## train

```sh
# pretrained:
python train.py --pretrained  --file filename

# not pretrained:
python train.py --no-pretrained
```

## error_test

* error_test direction should be like [test1 test2 test3 code1.txt code2.txt code3.txt]
* change array code1 code2 code3..... in error_test.py
* change names of test1 test2 test3 if your dir name is not the same as those in error_test.py
* run error_test.py


```sh
python error_test.py test_dir cut_result_dir
```

*error_test.gray.py is used for gray image* 

# cut

containes two function `cutTrainImages` `cutInputImages`

best.pkl -> 1 huawen 0 micro
last.pkl -> 0 huawen 1 micro