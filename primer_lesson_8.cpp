#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <exception>

using namespace std;

/*
程序错误和异常处理:
1、try块中不能抛出局部对象的指针，也就是对象的类型必须是可以复制的。
2、极端情况下，main()函数都可以被try块包含。
3、嵌套try，内层的try-catch永远不会捕获外层的try-chatch，反之确能。
4、自定义异常对象时，catch参数最好为引用参数。
5、匹配自定义类型参数：基类优先，所以派生类一般放在前面。匹配时会忽略const
6、在catch块中可以重新抛出，仅一个throw就可以，抛出已经有的对象，并不会复制。
7、抛出异常的函数 throw(可抛出异常列表)，注意定义指针时不能包括异常列表。
8、标准库异常都是std::exception派生类， bad_cast bad_alloc runtime_error 
*/
class A
{
public:
	A(int count=10) throw(bad_alloc) try
	{
	}
	catch (...)
	{
	}
};
void myTerminate()
{
	cout << "Uncaught exception!" << endl;
	exit(1);
}

class Trouble { //自定义异常类的标准格式    
public:
	Trouble(const char* pStr = "There's a problem"):pMessage(pStr){}
	const char* what() const throw() { return pMessage; }
private:
	const char* pMessage;
};

class MoreTrouble:public Trouble { //自定义继承异常类    
public:
	MoreTrouble(const char* pStr = "There's more trouble") :Trouble(pStr) {}
};

//定义可抛出异常的函数
void doThat(int argument) throw(Trouble, MoreTrouble)  //抛出异常函数定义开始
try
{

}
catch (...)
{

}

void (*function) (int) throw(Touble, Moretouble);  //正确
typedef void (*function) (int) throw(Touble, Moretouble);  //错误，因为异常不是类型的一部分

typedef void (*function) (int);   //正确
function pFunction throw(Touble, Moretouble)  //只能在声明中包含异常说明

//---------------------异常函数结束
int main()
{
	//set_terminate(myTerminate);  //C++98版本自定义意外结束方法，绕过abort()函数
	//terminate();
	int test = 56;
	A a;
	try
	{
		if (test == 1) throw test;
		if (test > 5) throw "test is greater than 5";
		if (test == 2) throw a;
		if (test == 3) throw runtime_error("Error reading the file.");
		cout << "End try block" << endl;
	}//try块与catch块之间不能有其它代码
	catch (const char* str)    //注意 抛出异常会忽略const 
	{
		cout << str << endl;
	}
	catch (const int& i)
	{
		cout << i << endl;
	}
	catch (...)               //如果以上没有捕获到，就被此捕获
	{
		cout << "other catch" << endl;
	}

	return 0;
}

