/*
1、运算符号优先级：括号>加减乘除>判断符号
2、位运算符号：按位与(&)按位或(|)按位异或(^)按位取反(~)左移(<<)右移(>>)
3、if判断语句  for循环语句  无限循环   双目运算符  ++ += /=
4、条件运算答：? :
5、while 和do-while循环语句 无限循环
6、switch 条件语句
7、跳出和中断
8、string字符串类型
9、数组(字符串数组)的遍历
*/


#include <iostream>
#include <string>

int main() {
    int a = 60;  // 二进制表示为 0011 1100
    int b = 13;  // 二进制表示为 0000 1101

    // 按位与运算
    int c1 = a & b;  // 结果为 12, 二进制表示为 0000 1100
    // 按位或运算
    int c2 = a | b;  // 结果为 61, 二进制表示为 0011 1101
    // 按位异或运算
    int c3 = a ^ b;  // 结果为 49, 二进制表示为 0011 0001
    // 按位取反运算
    int c4 = ~a;     // 结果为 -61, 二补数表示为 1100 0011
    // 左移运算
    int c5 = a << 2; // 结果为 240, 二进制表示为 1111 0000
    // 右移运算
    int c6 = b >> 1; // 结果为 6.5, 二进制表示为 0000 0110 (如果是无符号类型则结果为3)

    std::cout << "按位与: " << c1 << std::endl;
    std::cout << "按位或: " << c2 << std::endl;
    std::cout << "按位异或: " << c3 << std::endl;
    std::cout << "按位取反: " << c4 << std::endl;
    std::cout << "左移: " << c5 << std::endl;
    std::cout << "右移: " << c6 << std::endl;

    //for循环
    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
    }

    int init_i = 0;
    while (init_i < 10) {
        std::cout << init_i << std::endl;
        init_i++;
    }
    // while(True)  for(;;)   无限循环
    init_i = 0;
    do {
        std::cout << init_i << std::endl;
        init_i++;
    } while (init_i < 10);

    int day = 3;

    switch (day) {
    case 1:
        std::cout << "Monday" << std::endl;
        break;
    case 2:
        std::cout << "Tuesday" << std::endl;
        break;
    case 3:
        std::cout << "Wednesday" << std::endl;
        break;
    default:
        std::cout << "Another day" << std::endl;
    }
    /*string常用的函数：
        1.std::string::size() : 返回字符串的长度。
        2.std::string::length() : 返回字符串的长度。
        3.std::string::empty() : 检查字符串是否为空。
        4.std::string::clear() : 清空字符串。
        5.std::string::insert() : 在指定位置插入一个字符或字符串。
        6.std::string::erase() : 删除指定位置的字符或子字符串。
        7.std::string::replace() : 替换指定位置的字符或子字符串。
        8.std::string::append() : 在字符串末尾添加一个字符或字符串。
        9.std::string::compare() : 比较两个字符串。
        10.std::string::substr() : 返回字符串的指定子字符串。
        11.std::string::find() : 查找指定字符或字符串在字符串中首次出现的位置。
        12.std::string::rfind() : 查找指定字符或字符串在字符串中最后一次出现的位置。
        13.std::string::find_first_of() : 查找指定字符或字符串在字符串中首次出现的位置。
        14.std::string::find_last_of() : 查找指定字符或字符串在字符串中最后一次出现的位置。
        15.std::string::at() : 访问字符串中的字符。
     */  
    return 0;
}
