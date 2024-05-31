#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;
/*
C++中vector用法
向量（vector）：连续存储的元素
头文件包含：#include <vector>

Vector容器简介
(1)vector是将元素置于一个动态数组中加以管理的容器
(2)vector可以随机存取元素（支持索引值直接存取，用[]或at()方法）
(3)vector尾部添加或移除元素很快。但是在中间或头部插入、移除就很费时。

Vector对象的构造
① 默认构造：
使用默认构造函数构造出来的vector对象的size为0
vector<int> vec;
vector<float> vec;
vector<string> vec;
class A {};
vector<A*> vec;                        // 用于存放A对象指针的vector容器
vector<A> vec;                         // 用于存放A对象的vector容器

② 带参数构造:
vector(beg, end);                      // 构造函数将[beg, end)元素拷贝给自身
vector(n,elem);                        // 构造函数将n个elem拷贝给自身
vector(const vector& vec);             // 拷贝构造函数

③ Vector的赋值
vector.assign(beg, end);               // 将[beg, end)数据拷贝赋值给本身
vector.assign(n, elem);                // 将n个elem拷贝赋值给本身
vector& operator=(const vector& vec);  // 重载等号操作符
vector.swap(vec);                      // 将vec与本身的元素互换
使用vector.assign()时，会将vector中原本的元素都清空，再执行赋值操作

vector.size();                         //计算vector中的元素数量 
vector.begin();                        //指向容器中的第一个元素的迭代器（指针）
vector.end();                          //指向容器中的最后一个元素的下一位的迭代器（指针）,左闭右开

vector.size();                         // 返回容器中元素的个数
vector.empty();                        // 判断容器是否为空
vector.resize(num);                    // 重新指定容器的长度为num，若容器变长，则以默认值0填充新位置，如果容器变短，则末尾超出容器长度的元素被删除
vector.resize(num, elem);              // 重新指定容器的长度为num，若容器变长，则以elem填充新位置，如果容器变短，则末尾超出容器长度的元素被删除

下表法：访问vector容器中的元素，下标越界，可能会导致程序异常终止，且不会输出异常原因。但是使用STL中vector提供的访问方法就能够给出报错原因。
vector.at(idx); // 返回索引idx所指的数据，如果idx越界，抛出out_of_range异常
vec[idx]; // 返回索引idx所指的数据，越界运行会报错

⑤ vector的插入和 删除 （“增删”）
vector末尾添加和删除操作
vector<int> vec;
vec.push_back(1);
vec.push_back(2);
vec.pop_back();

vector的插入
vector.insert(pos, elem); // 在pos位置插入一个elem元素的拷贝，返回新数据的位置
vector.insert(pos, n, elem); // 在pos位置插入n个elem数据，无返回值
vector.insert(pos, beg, end); // 在pos位置插入[beg, end)区间的数据，无返回值
pos应该是指针或者迭代器，不能是下标
⑥ 打印vector中的所有元素
vector<int> vec = {1, 2, 3, 4, 5};
for (int i = 0; i < vec.size(); i++) {
	cout << vec[i] << " ";
}
⑦ vector容器中迭代器的基本使用
vector<int>::iterator iter;
vector容器的迭代器属于“随机访问迭代器”：迭代器一次可以移动多个位置

vector<int>::iterator iter = v.begin();
for (auto iter = v.begin(); iter != v.end(); iter++)
    *iter = 0;

*/
//vector<int> vec;
//vector<float> vec;
//vector<string> vec;
//class A {};
//vector<A*> vec; // 用于存放A对象指针的vector容器
//vector<A> vec; // 用于存放A对象的vector容器
vector<int> vec = { 1, 2, 3, 4, 5 };
int main()
{
	vec.capacity();
	cout << vec.capacity() << endl;
	return 0;
}

