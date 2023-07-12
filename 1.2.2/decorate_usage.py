# 装修器的基础用法，基础结构
def use_logging(func):
    def _deco():
        print(f"{func.__name__} is running")
        func()
    return _deco

@use_logging
def tool():
    print("I am tool")

tool()

# 装修器带固定参数
def use_logging(func):
    def _deco(a, b):
        print(f"{func.__name__} is running")
        func(a, b)
    return _deco

@use_logging
def tool(a, b):
    print(f"I am tool:{a+b}")

tool(1, 2)

# 非常实用的装饰线程函数
from threading import Thread
from time import sleep

def async_decor(func):
    def _deco(*args, **kwargs):
        thr = Thread(target = func, args=args, kwargs=kwargs)
        thr.start()
    return _deco

@async_decor
def A(a,b):
    sleep(4)
    print("a function",a+b)

def B(a, b): # 注意这个函数没有被装饰
    print("a function",a+b)

A(3, 5)
B(6, 7)

#装饰不确定长度的参数或类型
def use_logging(func):
    def _deco(*args, **kwargs):
        print(f"{func.__name__} is running")
        func(*args, **kwargs)
    return _deco

@use_logging
def tool(a, b):
    print(f"I am tool:{a+b}")

@use_logging
def food(a, b, c):
    print(f"I am food:{a+b+c}")

tool(1, 2)
food(2, 3, 4)

#带参数的装饰器
def use_logging(level):
    def _deco(func):
        def __deco(*args, **kwargs):
            if level == "warm":
                print(f"{func.__name__} is running")
                func(*args, **kwargs)
        return __deco
    return _deco

@use_logging(level="warm")
def tool(a, b):
    print(f"I am tool:{a+b}")

tool(1, 3)

#带参数和不带参数的自适应装饰器
from functools import wraps
def use_logging(arg=None):
    if callable(arg):    #  判断传入的参数是不是函数，用不带参数的装饰器调用该分支。
        @wraps(arg)
        def _deco(*args, **kwargs):
            print(f"{arg.__name__} is no parameter running")
        return _deco
    else:
        def _deco(func):
            @wraps(func)
            def __deco(*args, **kwargs):
                if arg == "warn":
                    print(f"{func.__name__} is passing parameter running")
                    func(*args, **kwargs)
            return __deco
        return _deco

def func():
    print("This is no parameter running")

@use_logging("warm")
#@use_logging(func())
def tool(a, b):
    print(f"I am tool:{a+b}")
    print(tool.__name__)

tool(11, 3)

# 类装饰器
class loging(object):
    def __init__(self, level="warn"):
        self.level = level
    def __call__(self, func):  # 关键重写方法,类方法，相当于C++仿函数
        def __deco(*args, **kwargs):
            if self.level == "warn":
                self.notify(func)
            return func(*args, **kwargs)
        return __deco
    def notify(self, func):  # 利用多态性调用
        print(f"{func.__name__} is running")

@use_logging("warn")
def tool(a, b):
    print(f"I am tool:{a+b}")

tool(11, 13)

#继承类装饰器
class email_loging(loging): # 继承上例的类
    def __init__(self, email='wymzj@163.com', *args, **kwargs):
        self.email = email
        #super(email_loging, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
    def notify(self, func):
        print(f"{func.__name__} is running")
        print(f"sending email to {self.email}")

@email_loging("warm")
def tool(a, b):
    print(f"I am tool:{a+b}")

tool(61, 13)

# 被装饰的类:单例类
def cl_log(instance=False):
    def dec(cls):
        instance = None
        def _deco(*args, **kwargs):
            nonlocal instance
            if instance is None:
                instance = cls(*args, **kwargs)
            return instance
        return _deco
    return dec

@cl_log(True)
class A:
    def __init__(self, *args, **kwargs):
        print(*args, **kwargs)

# 实例化装饰后的类
a = A("wym")
print(id(a))
b = A("lingyuntech")
print(id(b))

# 被装饰的类
def cl_log(cls):
    class dec(object):
        def __init__(self, *args, **kwargs):
            self.instance = cls(*args, **kwargs)
    return dec

@cl_log
class A:
    def __init__(self, *args, **kwargs):
        print(*args, **kwargs)

# 实例化装饰后的类
a = A("wym")
print(id(a))
b = A("lingyuntech")
print(id(b))

