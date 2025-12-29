#include <iostream>
#include <string>

int main() {
  // 1. 绑定字面常量（右值）
  int &&rval1 = 100;
  std::cout << "rval1: " << rval1 << std::endl; // 输出：100

  // 2. 绑定函数返回的临时对象（右值）
  std::string &&rval2 = std::string("hello") + " world";
  std::cout << "rval2: " << rval2 << std::endl; // 输出：hello world

  std::string lval = rval2;
  std::cout << "lval: " << lval << std::endl; // 输出：hello world

  // 3. 错误示例：右值引用不能直接绑定左值
  int a = 50;
  // 编译报错：无法将左值绑定到右值引用
  // int &&rval3 = a; // error: cannot bind rvalue reference of type ‘int&&’ to
  // lvalue of type ‘int’

  return 0;
}
