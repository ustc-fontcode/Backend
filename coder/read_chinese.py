import re


def read_chinese3500() -> str:
    fonts_range = []
    for lines in open('data/frequent_chars/chinese_u8.txt', 'r'):
        fonts_range.extend(map(lambda x: int(x, 16), re.split(',', lines.strip().strip(','))))

    letter_list = ''.join(chr(yf) for yf in fonts_range)
    return letter_list


def read_chinese3000() -> str:
    normal_chars = open('data/frequent_chars/frequent_chars.txt').read()
    return normal_chars


if __name__ == "__main__":
    l = read_chinese3500()
    print(len(l))
    print(type(l))
