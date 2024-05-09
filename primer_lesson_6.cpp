#include <iostream>

/*
面向对象开发：类的继承
1、类的派生
2、继承下的访问控制
3、继承下的副本构造函数
4、继承下的析构函数
5、重复的函数名
6、多重继承
7、重复的继承
8、虚继承
9、纯虚类
*/
using namespace std;
//基类
class Box
{
public:
	Box(double lv = 1.7, double wv = 1.3, double hv = 1.0)
		:length(lv),width(wv),height(hv)
	{

	}
	~Box()
	{

	}
	double get_v() { return length * width * height; }
protected:
	double length;
	double width;
	double height;
};

//继承子类
class Carton : public  Box
{
public:
	Carton(const char* pStr = "Cardboard")
	{
		m_pMaterial = new char[strlen(pStr + 1)];
		strcpy_s(m_pMaterial, strlen(pStr)+1, pStr);
	}
	~Carton()
	{

	}
	double get_v() { return length * width * height; }
private:
	char* m_pMaterial;
};

//继承子类
class Soft : public  Box
{
public:
	Soft(const char* pStr = "Softboard")
	{
		m_pMaterial = new char[strlen(pStr + 1)];
		strcpy_s(m_pMaterial, strlen(pStr) + 1, pStr);
	}

	Soft(const Soft& so):Box(so)    //副本构造函数
	{
	
	}

	~Soft()
	{

	}
	double get_v() { return length * width * height; }
private:
	char* m_pMaterial;
};

//多重继承
class Middle : public Soft, public Carton
{
public:
	Middle(const char* pStr = "Cardboard")
	{
		m_pMaterial = new char[strlen(pStr + 1)];
		strcpy_s(m_pMaterial, strlen(pStr) + 1, pStr);

	}
	~Middle()
	{

	}
private:
	char* m_pMaterial;
};

//纯虚类
class AbstractClass {
public:
	virtual void interfaceFunction() = 0;
	// 可以有多个纯虚函数
	virtual void anotherInterfaceFunction() = 0;
	// 类可以包含成员变量和成员函数
	int commonVariable;
	void commonFunction();
};

int main()
{
	Box box(2.3,3.3,3.4);
	Middle mid("不软不硬");
	mid.Soft::Box::get_v();
	return 0;
}

//diameter
