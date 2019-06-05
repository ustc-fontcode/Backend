import bchlib
import reedsolo
from PIL import Image
from coder import generator
from coder import config
from coder import read_chinese


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
    '''
        普通BCH 编码 最多可以编码48 bits, 允许6bits error 
        但是没有解码检验
        需要时用Coder()进行初始化
        encode(code, text) 函数 接受code(待编码的字符串 如'baidu'),
                                   text(文档内容, 如"中华宪法...", 注意这里的文档字数需要 大于 编码后bits 位数)
                                输出 PIL.Image.Image
        decode(bits_list) 函数 接受bits_list(一个有int 0 1 组成的 list)
                               输出 解码之后的字符串
    '''

    def __init__(self):
        self._BCH_POLYNOMIAL = 8219
        self._BCH_ERROR_BITS = 6  # 编码48bits 允许6bits error 总长128bits
        self.core = bchlib.BCH(self._BCH_POLYNOMIAL, self._BCH_ERROR_BITS)

    def encode(self, code: str, text: str) -> Image.Image:
        assert (len(code) <= 6)
        code_bytes = bytearray(code, 'ascii')  # 48bits
        code_masked = code_bytes + self.core.encode(code_bytes)  # 80bits
        code_masked_bits_list = Converter.bytearray2bits_list(code_masked)
        text = text[:len(code_masked_bits_list)]
        docs = generator.generate_doc_with_code_and_bias(
            code_masked_bits_list, text, [
                "data/fonts/" + config.FONT_NAME_HuaWen + ".ttf",
                "data/fonts/" + config.FONT_NAME_Micro + ".ttf"
            ], {0: (0, -8)})
        docs[0].show()
        return docs[0]

    def decode(self, bits_list: list) -> str:
        # TODO: 如何判断解码是否正确
        bytes_array = Converter.bits_list2bytearray(bits_list)
        bitflips, code, ecc = self.core.decode(bytes_array[:-10],
                                               bytes_array[-10:])
        return code.decode('ascii')


class MyRS:
    '''
        使用方法和MyBCH 一样, 最多可以输入48 bits
        但是最多只能有5bits error
        但是有解码检测, 即告诉使用者是否解码正确, 如果不正确 decoder 返回''
    '''

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


if __name__ == "__main__":
    a = 'yjwhhh'
    coder = MyRS()
    coder.encode(a, read_chinese.read_chinese3000())
    print(
        coder.decode([
            1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0,
            1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
            0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,
            0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
        ]))
