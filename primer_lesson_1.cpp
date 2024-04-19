/*
初级第一节：变量
整形：int long
浮点型：float double
指针(地址)
字面量
具名变量
字符型
枚举型
引用变量
固定变量
静态变量
结构及初始化
数组
变量命名规范
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
int int_value = 10;               //短整形
long long_value = 100;            //长整型
float float_value = 3.14;         //单浮点型
double double_value = 3.1415926;  //双浮点型
char str = 'C';                   //字符变量
const int const_int = 10;         //固定变量
static int static_int = 0;        //静态变量
int& refer_int = int_value;       //引用变量，已有变量的别名
enum enumType {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday
};                                            //枚举变量 enumType  et = Monday

int* pValue = &int_value;
int* pInt = (int*)malloc(sizeof(int));        //C形式堆中分配内存
long* plong = new long;                       //C++形式分配内存
const char* pStr = "myfriend\0";
char char_array[5] = { 'a','b','c','d','e'};  //静态分配数组 
char char_array_a[] = { 'a','b','c','d','e'};  //静态分配数组
int int_array[10] = { 1,2,3,4,5,6,7,8,9,0};
int int_array_b[3][2] = {{2, 3},{4, 5},{5, 7}};
int* pNewInt = new int[10];                   //动态分配数组

struct pet
{
    int dog;
    char str;
    float cat;
    int* pDog;
    string mystr;
};                                            //结构类型

void main()
{
    free(pInt);
    enumType  et = Monday;
	cout << pStr << endl;
    printf("%d",long_value);
    int* p = int_array;
    for (int i = 0; i < 10; i++)
    {
        p++;
        cout << *p<< endl;
    }
    pet my_pet = {0};
    my_pet.mystr = "abc";
    
    int** ptr = new int*[10];   //二维指针
    for (int i = 0; i < 10; i++)
    {
        ptr[i] = new int[10];
    }
    delete[] * ptr;
    delete[] ptr;
}
