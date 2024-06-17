判断语句
第一种
```
if (判断条件)
{
  //语句
}
```

//第二种
```
if (判断条件)
{
  //语句
}
else 
{
  //语句
}
```

//第三种
```
if (判断条件)
{
  //语句
}
else if (判断条件)
{

}
else 
{
  //语句
}
```

for循环
```
for (int i = 0; i < 10; i++) {
    std::cout << i << std::endl;
}

for (int i = 10; i > 0; i--) {
    std::cout << i << std::endl;
}

for (int i = 0; i < 10; i=i+2) {
    std::cout << i << std::endl;
}
```
while循环语句：
在while循环中有时会有continue立即执行新的循环和break跳出循环语句。for(;;)等同于while(True)无限循环 
```
int init_i = 0;
while (init_i < 10) {
    std::cout << init_i << std::endl;
    init_i++;
}

init_i = 0;
do {
    std::cout << init_i << std::endl;
    init_i++;
} while (init_i < 10);  //至少执行一次,注意后面有个分号结束。
```

条件选择语句
```
int day = 3;
switch (day) {
case 1:
    std::cout << "Monday" << std::endl;
    break;   //break必须加上，否则会执行下面的代码。
case 2:
    std::cout << "Tuesday" << std::endl;
    break;
case 3:
    std::cout << "Wednesday" << std::endl;
    break;
default:     //如果其它条件都不满足就执行该条件。
    std::cout << "Another day" << std::endl;
}
```

1、数据类型：<br>
Int 短整型<br>
float 点浮点小数<br>
double 双浮点小数<br>
bool 布尔型变量【true非0｜false  0】<br>
char 字符类型<br>
const int 固定短整型变量<br>
unsigned int 无符号类型短整型，也就是没有正负之分<br>

2、多行注释符
```
/*
   注释语句1
   注释语句2
*/
```

3、控制语句：
①if判断：
```
if(判断条件)
{
    //语句
}

if(判断条件)
{
    //语句
}
else
{
    //语句
}

if(判断条件)
{
    //语句
}
else if(判断条件)
{
    //语句
}
else
{
    //语句
}
```

②三目条件运算：
```
int v = a>b ? 1 : 2    //如果a>b就选择1，否则为2
```

③for循环:
```
for(int I = 0; i< 10;i++)
{
	break;  //可以跳出for循环
	//其它语句
｝

for(int i = n; i > 0; i--) //也可以从大到小
{
	//语句
｝

for( ; i> 0 ; i--) //第一分号前可以为空
{
	//语句
｝

for(int i=n ; i>0 ; i--, j--) //可以多个变量迭代
{
	//语句
｝

for( ; ; )//可以都为空，与while(true)等价

```
④while循环
```
while(判断条件)
{
	continue; //忽略下面语句直接循环
    break；  //直接跳出循环
}

while(A=B+C)     //用B+C后得到的A值进行判断
{
	continue; //忽略下面语句直接循环
    break；  //直接跳出循环
}

do
{
	//语句
} while(判断条件)；//至少执行一次，注意后面有个分号
```

⑤switch语句：
```
switch (day) {
    case 1:
        //语句
        break;  //必须要加，否则会继续执行
    case 2:
        //语句
        break;
    case 3:
        //语句
        break; 
    default:  //如果以上条件都不匹配就默认执行
        //语句
}
```

4、自定义结构：
```
struct  my_s
{
    int a;
    float b;
    char str[100];
}

my_s  s = {0};  //{0}对结构进行初始化
s.a = 10;
s.str[2] = ‘B’;    //通过”.”号引出成员

my_s* ps = &s;  //得到结构变量的地址
s->a = 10;
s->str[2] = ‘B’;   //指针通过”->”号引出成员
```

5、宏
```
#define PI 3.1425926  //定义一个宏替换
```

6、其它关键字：
```
void   //函数无返回值
void*  //任何类型的地址
\n     //换行
%      //1、取余 2、格式化字符串
||      //条件或(只要有一个条件满足)
&&    //条件与(所有条件都满足) 
```
