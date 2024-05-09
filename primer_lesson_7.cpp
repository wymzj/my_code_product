#include <iostream>
/*
类的多态性：
1、虚函数对应继承顺序动态调用
2、虚函数默认参数
3、虚函数指针调用
4、虚函数引用调用
5、指针和继承类对象之间的转换
6、动态强制转换
7、多态性的成本
8、类成员的指针
*/
using namespace std;
class Box
{
public:
	Box(double lv = 1.7, double wv = 1.3, double hv = 1.0)
		:length(lv), width(wv), height(hv)
	{

	}
	~Box()
	{

	}
	virtual double get_v() { 
		cout << " this is base Box methon" << endl;
		return length * width * height;
	}
//protected:
public:
	double length;
	double width;
	double height;
};

//继承子类
class Carton : public  virtual Box
{
public:
	Carton(const char* pStr = "Cardboard")
	{
		m_pMaterial = new char[strlen(pStr + 1)];
		strcpy_s(m_pMaterial, strlen(pStr) + 1, pStr);
	}
	~Carton()
	{

	}
	double get_v(double l = 0, double w = 0, double h = 0)
	{
		cout<<" this is a Carton method" << endl;
		return length * width * height;
	}
private:
	char* m_pMaterial;
};

//继承子类
class Soft : public  virtual Box
{
public:
	Soft(const char* pStr = "Softboard")
	{
		m_pMaterial = new char[strlen(pStr + 1)];
		strcpy_s(m_pMaterial, strlen(pStr) + 1, pStr);
	}

	Soft(const Soft& so) :Box(so)    //副本构造函数
	{
	}

	~Soft()
	{
	}

	double get_v() 
	{
		cout << " this is a Soft method" << endl;
		return length * width * height; 
	}
private:
	char* m_pMaterial;
};

//多重继承
class Middle : public Soft, public Carton
{
public:
	Middle(const char* pStr = "Middleboard")
	{
		m_pMaterial = new char[strlen(pStr + 1)];
		strcpy_s(m_pMaterial, strlen(pStr) + 1, pStr);
	}
	~Middle()
	{

	}

	double get_v()
	{
		cout << " this is a Middle method" << endl;
		return this->Soft::length * this->Soft::width * this->Soft::height;
	}
private:
	char* m_pMaterial;
};

double reference_call(Box& b)          //引用虚函数调用
{
	cout << " this is a no const method" << endl;
	return b.get_v();
}
double reference_call(const Box& b)    //引用虚函数调用
{
	cout << " this is a const method" << endl;
	Box nb = const_cast<Box&>(b);      //去掉const类型转换
	return nb.get_v();
}
int main()
{
	Box* pB = new Soft;                        //隐式指针转换
	Box* pBs = static_cast<Box*>(new Soft);    //显示强制转换
	pB->get_v();                               //虚表动态调用     
	pB->Box::get_v();                          //指定基类作用域调用
	Middle mid;
	Box* pM = dynamic_cast<Box*>(& mid);       //显式指针转换
	pM->get_v();
	Carton ca;
	reference_call(ca);                        //通过引用值虚函数多态性调用
	reference_call(mid);

	double Box::* pData;                       //类成员变量指针
	Box box;
	pData = &Box::width;                       //width在此必须是全局变量
	cout << pB->*pData << endl;
	cout << mid.*pData << endl;                //类成员指针的运用

	double (Box::*pGet)();                     //类成员函数指针
	pGet = &Box::get_v;                        //类成员函数指针绑定具体方法         
	cout << (pB->*pGet)() << endl;
	cout << (mid.*pGet)() << endl;             //类成员方法的运用
	return 0;
}