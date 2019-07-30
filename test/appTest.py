import mimetypes
import unittest
from fileinput import filename

import requests
from flask import make_response

import app
import json
import urllib3

from coder.read_chinese import read_chinese3000


class TestApp(unittest.TestCase):
    """定义测试案例"""
    # 测试代码执行之前调用 (方法名固定)
    def setUp(self):
        self.http = urllib3.PoolManager()


    # 测试代码。 (方法名必须以"test_"开头)
    def test_generate(self):
        response = self.http.request("POST",
                                url="http://127.0.0.1:5000/generate",
                                fields=
                                {
                                    "user": "huangzp",
                                    "bits": "0101",
                                    "text": read_chinese3000()
                                }
                                )
        print(response.data)
        with open("test.png", "wb") as f:
            f.write(response.data)




if __name__ == '__main__':
    unittest.main()  # 进行测试
