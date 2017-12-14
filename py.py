#!/usr/bin/env python3
try:
    1 / 0
except ZeroDivisionError:
    print("err")
finally:
    print("finally")


class Demo(object):
    def __enter__(self):
        print("enter")

    def __exit__(self, type_arg, value_arg, traceback_arg):
        print(type_arg)
        print(value_arg)
        print(traceback_arg)
        print("exit")
        return True


with Demo():
    print("tt")
    1 / 0
