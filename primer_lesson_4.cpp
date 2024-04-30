/*
1、字节对齐
2、联合，匿名联合
3、内存操作函数：memset memcpy
*/
#include <iostream>
#include <stdio.h> 
#include <string.h>


struct {              //匿名结构  
    char name[40];
    int age;
} person, person_copy;

struct alignas(32) AligStruct //字节对齐 在标准位的基础上扩展用。在GCC下一般用memalign等函数
{
    char name[63];
    alignas(1) int age;
    int salary;
} alstru;

union MyUnion          //声明联合
{
    double ShareID;
    long LVal;
};

union {
    double ShareID;
    long LVal;
} uvalue;

//参考预编译指令进行字节对齐
//#pragma pack(push, 16)
_Pragma("pack(1)")//表示它后面的代码都按照1个字节对齐，在标准位的基础上缩小
struct stru3
{
    char name[40];
    int age;
} stru3;
_Pragma("pack()")//取消按照n个字节对齐，是对#pragma pack(n)的一个反向操作
//#pragma pack(pop)
int main()
{
    printf("结构所占字节：%d\n", sizeof(alstru));
    printf("结构所占字节：%d\n", sizeof(stru3));
    char str[50];
    memset(str, 0, sizeof(str));
    strcpy_s(str, "This is string.h library function");
    printf("%s\n", str);
    memset(str, '$', 7);
    printf("%s\n", str);

    char myname[] = "Linyuntech";
    /* 使用内存拷贝字符串*/
    memcpy(person.name, myname, strlen(myname) + 1);
    person.age = 46;
    /* 使用内存拷贝结构 */
    memcpy(&person_copy, &person, sizeof(person));

    return(0);
}
