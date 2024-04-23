/*
面向对象开发:
1、类的封装性
2、构造函数
3、副本构造函数
4、析构函数
5、友元函数:如果某个函数不是某个类的成员，但可以访问类的所有成员，这个函数就叫友元函数。
6、友元类：一个类声明为另一个类的友元，友元类的成员函数不受限制的访问原类的成员。
*/


#include <iostream>
#include "Class_learn.h"

class A 
{
public:
    int data;

    A(int value=0):data(value)   //构造函数
    {
        //this->data = value;
        std::cout << data << " construct function\n"<<std::endl;
    }

    A(const A& a)  //副本构造函数
    {
        this->data = a.data;
        std::cout << data << " copy construct function\n" << std::endl;;
    }

    ~A()   //析构函数
    {
        std::cout << data << " desconstruct function\n"<< std::endl;
    }

    A& operator=(const A& other)
    {
        this->data = other.data;
        std::cout << " operator=\n" << std::endl;
        return *this;
    }

    A operator+(const A& c) const {
        return c.data + this->data;
    }

    friend std::ostream& operator<<(std::ostream& out, const A& c) {
        out << "(" << c.data << " + " << c.data+1 << ")";
        return out;
    }
    friend int get_age(const A& a);  //友元函数 
    friend class Class_learn;        //将这个类的私有成员开放给类Class_learn
private:
    int age;    //私有成员
};
int get_age(const A& a) { return a.age; }
A operator+(const A& a, const A& b)
{
    return a.data + b.data;
}

int main()
{
    A a(22);
    get_age(a);
    A A_one = A(10);
    std::cout << A_one<<std::endl;
    A* pA_two = new A(11);
    delete pA_two;
    A A_three = A(A_one);
    A A_four = A(12);
    A_four = A_three;
    return 0;
};

