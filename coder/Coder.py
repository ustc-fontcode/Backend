from random import random
from time import sleep

import bchlib
import reedsolo
from PIL import Image

from coder import config, read_chinese
from coder import generator


class Converter:
    def bytearray2bits_list(the_bytes: bytearray) -> list:
        bytes_list = list(the_bytes)
        return [
            int(x) for x in ''.join(
                format(x, '#010b')[2:] for x in bytes_list)
        ]

    def bits_list2bytearray(bits_list: list) -> bytearray:
        bytes_list1 = [
            bits_list[i * 8:i * 8 + 8] for i in range(len(bits_list) // 8)
        ]
        bytes_list2 = [
            int(''.join(str(x) for x in byte_list), 2)
            for byte_list in bytes_list1
        ]
        # bytes_list3 = ''.join(chr(x) for x in bytes_list2)
        return bytearray(bytes_list2)


class MyBCH:
    """
        普通BCH 编码 最多可以编码48 bits, 允许6bits error
        但是没有解码检验
        需要时用Coder()进行初始化
        encode(code, text) 函数 接受code(待编码的字符串 如'baidu'),
                                   text(文档内容, 如"中华宪法...", 注意这里的文档字数需要 大于 编码后bits 位数)
                                输出 PIL.Image.Image
        decode(bits_list) 函数 接受bits_list(一个有int 0 1 组成的 list)
                               输出 解码之后的字符串
    """

    # p = 8219
    def __init__(self, p=451, t=15):
        self._BCH_POLYNOMIAL = p
        self._BCH_ERROR_BITS = t  # 编码48bits 允许6bits error 总长128bits
        self.core = bchlib.BCH(self._BCH_POLYNOMIAL, self._BCH_ERROR_BITS)

    def encode(self, code: str, text: str) -> "list[Image.Image]":
        code_bytes = bytearray(code, 'ascii')  # 80bits
        code_masked = code_bytes + self.core.encode(code_bytes)  # 80bits
        self._e = (len(code_masked) - len(code_bytes)) * 8
        code_masked_bits_list = Converter.bytearray2bits_list(code_masked)
        self._hint = code_masked_bits_list
        text = text[:len(code_masked_bits_list)]
        docs = generator.generate_doc_with_code_and_bias(
            code_masked_bits_list, text, [
                "data/fonts/" + config.FONT_NAME_HuaWen + ".ttf",
                "data/fonts/" + config.FONT_NAME_Micro + ".ttf"
            ], {0: (0, -8)})
        return docs

    def decode(self, bits_list: list) -> str:
        # TODO: 如何判断解码是否正确
        # bytes_array = Converter.bits_list2bytearray(bits_list)
        tmp = self._hint
        e = self._BCH_ERROR_BITS
        tmp[: e] = [(1 - x) for x in tmp[: e]]
        bytes_array = Converter.bits_list2bytearray(tmp)

        bitflips, code, ecc = self.core.decode(bytes_array[: - self._e // 8],
                                               bytes_array[- self._e // 8:])
        return code.decode('ascii')


class MyRS:
    """
        使用方法和MyBCH 一样, 最多可以输入48 bits
        但是最多只能有5bits error
        但是有解码检测, 即告诉使用者是否解码正确, 如果不正确 decoder 返回''
    """

    def __init__(self):
        self.core = reedsolo.RSCodec(10)

    def encode(self, code: str, text: str) -> Image.Image:
        assert (len(code) <= 6)
        code_bytes = bytearray(code, 'ascii')  # 48bits
        print(code_bytes)
        code_masked = self.core.encode(code_bytes)  # 80bits
        print(code_masked)
        code_masked_bits_list = Converter.bytearray2bits_list(code_masked)
        print(code_masked_bits_list)
        print(len(code_masked_bits_list))
        text = text[:len(code_masked_bits_list)]
        docs = generator.generate_doc_with_code_and_bias(
            code_masked_bits_list, text, [
                "data/fonts/" + config.FONT_NAME_HuaWen + ".ttf",
                "data/fonts/" + config.FONT_NAME_Micro + ".ttf"
            ], {0: (0, -8)})
        docs[0].show()
        return docs[0]

    def decode(self, bits_list: list) -> str:
        bytes_array = Converter.bits_list2bytearray(bits_list)
        try:
            code = self.core.decode(bytes_array)
            result = code.decode('ascii')
        except reedsolo.ReedSolomonError:
            result = ''
        return result


class MyVote:
    def __init__(self, c=4, total=144):
        """
        :param c: c表示编码c bits 信息 默认 4
        :param total: 总共的编码长度 默认 144
        """
        self.total = total
        self.c = c
        self.block_size = total // c

    def encode(self, data: list) -> list:
        """

        :param data: data 就是长为self.c (4) 的待编码数据
        :return: 长为 self.total (144) 的编码结果
        """
        assert len(data) == self.c
        result = [0] * self.total
        for i in range(self.c):
            result[self.block_size * i: self.block_size * (i + 1)] = [data[i]] * self.block_size

        return result

    def decode(self, data: list) -> list:
        """

        :param data: 长为self.total (144) 待解码数据
        :return: 长为self.c (4) 的 解码数据
        """
        assert len(data) == self.total
        result = [0] * self.c
        for i in range(self.c):
            result[i] = round(sum(data[self.block_size * i: self.block_size * (i + 1)])
                              / self.block_size)
        return result


if __name__ == "__main__":
    # a = "hh"
    # p_available = []
    # for p in range(1000):
    #     try:
    #         coder = MyBCH(p=p, t=6)
    #     except RuntimeError:
    #         continue
    #     p_available.append(p)
    # print(p_available)
    # sleep(10)
    # for p in p_available:
    #     print("p:", p)
    #     coder.encode(a, "黄志鹏" * 100)
    #     print(len(coder._hint))
    test_epochs = 100
    test_error = 70
    data = [0, 1, 0, 1]
    coder = MyVote()
    code = coder.encode(data)
    count = 0
    for i in range(test_error):
        for j in range(test_epochs):
            test_code = code.copy()
            idxs = [0] * i
            for s in range(i):
                idxs[s] = min(coder.total - 1, round(random() * coder.total))
            for idx in idxs:
                test_code[idx] = 1 - test_code[idx]

            re_data = coder.decode(test_code)
            if re_data != data:
                count += 1
    print("error_rate", count / (test_epochs * test_error))
