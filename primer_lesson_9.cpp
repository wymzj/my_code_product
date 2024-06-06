#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

1.C++中vector用法

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

2.C++中list用法
#include <iostream>
#include <list>
using namespace std;

int main()
{
    list<int> lt;
    // 头插数据
    lt.push_front(1);
    lt.push_front(2);
    lt.push_front(3);
    for (auto e : lt)
    {
        cout << e << " ";
    }
    cout << endl;
    // 头删数据
    lt.pop_front();
    for (auto e : lt)
    {
        cout << e << " ";
    }
    cout << endl;
    return 0;
}

#include <iostream>
#include <list>
using namespace std;

int main()
{
    list<int> lt;
    // 尾插数据
    lt.push_back(1);
    lt.push_back(2);
    lt.push_back(3);
    for (auto e : lt)
    {
        cout << e << " ";
    }
    cout << endl;
    // 尾删数据
    lt.pop_back();
    for (auto e : lt)
    {
        cout << e << " ";
    }
    cout << endl;
    return 0;
}

int main()
{
	list<int> lt;
	lt.push_back(1);
	lt.push_back(2);
	lt.push_back(3);
	list<int>::iterator pos = find(lt.begin(), lt.end(), 2);
	lt.insert(pos, 4); //在2的位置插入4
	for (auto e : lt)
	{
		cout << e << " ";
	}
	cout << endl; 

	pos = find(lt.begin(), lt.end(), 3);
	lt.insert(pos, 3, 5); //在3的位置插入3个5
	for (auto e : lt)
	{
	    cout << e << " ";
	}
	cout << endl;

	vector<int> v{ 6, 6 };
	pos = find(lt.begin(), lt.end(), 1);
	lt.insert(pos, v.begin(), v.end()); //在1的位置插入2个6
	for (auto e : lt)
	{
	    cout << e << " ";
	}
	cout << endl;
	return 0;
}

int main()
{
	list<int> lt;
	lt.push_back(1);
	lt.push_back(2);
	lt.push_back(3);
	lt.push_back(4);
	list<int>::iterator pos = find(lt.begin(), lt.end(), 2);
	lt.erase(pos); // 删除2
	for (auto e : lt)
	{
		cout << e << " ";
	}
	cout << endl;

	pos = find(lt.begin(), lt.end(), 3);
	lt.erase(pos, lt.end()); //删除3及其之后的元素
	for (auto e : lt)
	{
		cout << e << " ";
	}
	cout << endl;
	return 0;
}

list迭代器的使用

#include <iostream>
#include <list>

using namespace std;

int main()
{
    string s("hello");
    list<char> lt(s.begin(), s.end());
    //正向迭代器遍历容器
    list<char>::iterator it = lt.begin();
    while (it != lt.end())
    {
        cout << *it << " ";
        it++;
    }
    cout << endl;

    //反向迭代器遍历容器
    list<char>::reverse_iterator rit = lt.rbegin();
    while (rit != lt.rend())
    {
        cout << *rit << " ";
        rit++;
    }
    cout << endl;
    return 0;
}

2.map用法
my_map.insert()或按照数组直接赋值：插入
my_map.find()：查找一个元素
my_map.clear()：清空
my_map.erase()：删除一个元素
my_map.size()：map的长度大小
my_map.begin()：返回指向map头部的迭代器
my_map.end()：返回指向map末尾的迭代器
my_map.rbegin()：返回一个指向map尾部的逆向迭代器
my_map.rend()：返回一个指向map头部的逆向迭代器
my_map.empty()：map为空时返回true
swap()：交换两个map，两个map中所有元素都交换

用insert函数插入pair数据：

map<int, string> my_map;
my_map.insert(pair<int, string>(1, "a"));

用insert函数插入value_type数据:

map<int,string> my_map;
my_map.insert(map<int,string>::value_type(1,"first"));
my_map.insert(map<int,string>::value_type(2,"second"));
 
map<int,string>::iterator it;           //迭代器遍历
for(it=my_map.begin();it!=my_map.end();it++)
    cout<<it->first<<it->second<<endl;

用数组的方式直接赋值:

map<int, string> my_map;
my_map[1] = "first";

查找元素（判定这个关键字是否在map中出现）
用count函数来判断关键字是否出现，其缺点是无法定位元素出现的位置。由于map一对一的映射关系，count函数的返回值要么是0，要么是1

map<string, int> my_map;
my_map["first"] = 1;
cout << my_map.count("first") << endl;

用find函数来定位元素出现的位置，它返回一个迭代器，当数据出现时，返回的是数据所在位置的迭代器；若map中没有要查找的数据，返回的迭代器等于end函数返回的迭代器。

#include <map>  
#include <string>  
#include <iostream>  
 
using namespace std;  
  
int main()  
{  
    map<int, string> my_map;  
    my_map.insert(pair<int, string>(1, "student_one"));  
    my_map.insert(pair<int, string>(2, "student_two"));  
    my_map.insert(pair<int, string>(3, "student_three"));  
    map<int, string>::iterator it;  
    it = my_map.find(1);  
    if(it != my_map.end())  
       cout<<"Find, the value is "<<it->second<<endl;      
    else  
       cout<<"Do not Find"<<endl;  
    return 0;  
}
//通过map对象的方法获取的iterator数据类型是一个std::pair对象，包括两个数据iterator->first和iterator->second，分别代表关键字和value值。

删除元素
#include <map>  
#include <string>  
#include <iostream>  
  
using namespace std;  
  
int main()  
{  
    map<int, string> my_map;  
    my_map.insert(pair<int, string>(1, "one"));  
    my_map.insert(pair<int, string>(2, "two"));  
    my_map.insert(pair<int, string>(3, "three"));  
    //如果你要演示输出效果，请选择以下的一种，你看到的效果会比较好
    //如果要删除1,用迭代器删除
    map<int, string>::iterator it;  
    it = my_map.find(1);  
    my_map.erase(it);                   //如果要删除1，用关键字删除
    int n = my_map.erase(1);            //如果删除了会返回1，否则返回0
    //用迭代器，成片的删除
    //一下代码把整个map清空
    my_map.erase( my_map.begin(), my_map.end() );  
    //成片删除要注意的是，也是STL的特性，删除区间是一个前闭后开的集合
    //自个加上遍历代码，打印输出吧
    return 0;
}  

排序，按value排序
map中元素是自动按key升序排序（从小到大）的；按照value排序时，想直接使用sort函数是做不到的，sort函数只支持数组、vector、list、queue等的排序，无法对map排序，那么就需要把map放在vector中，再对vector进行排序。

#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
using namespace std;
 
bool cmp(pair<string,int> a, pair<string,int> b) {
	return a.second < b.second;
}
 
int main()
{
    map<string, int> ma;
    ma["Alice"] = 86;
    ma["Bob"] = 78;
    ma["Zip"] = 92;
    ma["Stdevn"] = 88;
    vector< pair<string,int> > vec(ma.begin(),ma.end());
    //或者：
    //vector< pair<string,int> > vec;
    //for(map<string,int>::iterator it = ma.begin(); it != ma.end(); it++)
    //    vec.push_back( pair<string,int>(it->first,it->second) );
 
    sort(vec.begin(),vec.end(),cmp);
    for (vector<pair<string,int>>::iterator it = vec.begin(); it != vec.end(); ++it)
    {
        cout << it->first << " " << it->second << endl;
    }
    return 0;
}
MAP优点：
有序性，这是map结构最大的有点，其元素的有序性在很多应用中都会简化很多的操作
红黑树，内部实现一个红黑树使得map的很多操作在logn的时间复杂度下就可以实现，因此效率非常的高
缺点：
空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间
适用处：对于那些有顺序要求的问题，用map会更高效一些

MAP和unordered_map各自优缺点	
优点：因为内部实现了哈希表，因此其查找速度非常的快
缺点：哈希表的建立比较耗费时间
适用处：对于查找问题，unordered_map 会更加高效一些，因此遇到查找问题，常会考虑一下用unordered_map

创建自己的迭代器
#include <iostream>

using namespace std;

class Integer
{
public:
	Integer(int arg = 0) :x(arg) {}
	bool operator !=(const Integer& arg) const {
		return x != arg.x;
	}
	int operator*() const { return x; }

	Integer& operator++() {
		++x;
		return *this;
	}

	const Integer operator++(int) {
		Integer temp(*this);
		++x;
		return temp;
	}
private:
	int x;
};

template <typename Iter>
double average(Iter a, Iter end) {
	double sum = 0.0;
	int count = 0;
	for (; a != end; ++count)
		sum += *a++;
	return sum / count;
		
}

int main() {
	Integer first(1);
	Integer last(11);
	cout << "The average of the integers from" << *first << " to " << *last - 1;
	cout << " is " << average(first, last) << endl;
}


