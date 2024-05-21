/*
函数编程：
1、函数的声明
2、给函数传参数，按值传递、按引用传递、默认参数值
3、返回值，返回一个指针，返回一个引用，返回一个新变量
4、内联函数
5、函数重载
6、函数指针
7、递归函数
8、模板函数
9、函数指针
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

double pwoer(double x, int n=0)
{
	double dx = x + 3.14;
	int dy = n + 2;
	return dx + dy;
}

int substract(int& x, int& y) {	return x - y;}
int substract(int* px, int* py) { return *px - *py; }
//int substract(const int x, const int y) { return x + y; }
int substract(const int x, const int y, int z) { return x + y + z; }

double average(double array[], int count)
{
	double sum = 0.;
	for (int i = 0; i < count; i++)
	{
		sum += array[i];
	}
	return sum/count;
}

//double average(double* pArray, int count)
//{
//	double sum = 0.;
//	for (int i = 0; i < count; i++)
//	{
//		sum += *pArray++;
//	}
//	return sum / count;
//}

int* backpoint(int a)  //(int* a) 正确
{
	return &a; //错误
	//return a;
}

int add(int x)         //函数中的静态变量
{
	static int sum = 0;
	sum += x;
	return sum;
}

inline int larger(int m, int n)  //内联函数
{
	return m>n?m:n;
}

int recurence(int n)            //递归函数
{
	if (n < 0)
	{
		return n; 
	}
	else
	{
		n -= 1;
		recurence(n);
	}
}

class Greater      //仿函数
{
public:
	bool operator()(int a, int b)
	{
		return a > b;
	}
}
/*匿名函数
[捕获列表](参数列表) mutable(可选) 异常属性 -> 返回类型 
{
   // 函数体
}
[=, &x, &，this]  //捕获列表 
*/
auto f = [] (int x, int y) mutable throw()|noexcept  -> int {return x + y;}

void value_capture() {
    int value = 1;
    auto copy_value = [value] {
        return value;
    };
    value = 100;
    auto stored_value = copy_value();
    std::cout << "stored_value = " << stored_value << std::endl;
}

template<class T>               //模板函数
T template_fun(T a, T b) { return a + b; }

int (*pfun)(int*, int*);       //函数指针

int func(int (*pfun)(int*, int*), int* a,int* b)
{
	return pfun(a, b);
}
int main()
{
	double double_arr[] = { 2,3,4,5,6 };
	average(double_arr, sizeof(double_arr) / sizeof(double_arr[0]));
	int a = 8, b = 9;
	substract(a, b);
	pfun = substract;
	pfun(&a, &b);
	func(substract, &a, &b);
	int m = 0, n = 0;
	[&, n] (int a) mutable { m = ++n + a; }(4);
	cout << m << endl << n << endl;
	return 0;
}
