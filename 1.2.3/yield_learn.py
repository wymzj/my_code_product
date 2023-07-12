# class Fab(object):
#     def __init__(self, max):
#         self.max = max
#         self.n, self.a, self.b = 0, 0, 1
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.n < self.max:
#             r = self.b
#             self.a, self.b = self.b, self.a + self.b
#             self.n += 1
#             return r
#         raise StopIteration()

def Fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n += 1

for n in Fab(5):
    print(n)


def generate_iter():
    print("beginning period")
    while True:
        result = yield 'output'
        print("result:", result)

it = generate_iter()
print(next(it))
print(next(it))