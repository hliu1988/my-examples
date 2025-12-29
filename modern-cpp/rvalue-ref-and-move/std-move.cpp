#include <iostream>
#include <utility> // 包含std::move
#include <vector>

int main() {
  // 定义一个大型vector（左值）
  std::vector<int> big_vec = {1, 2, 3, 4, 5};
  std::cout << "big_vec 初始大小：" << big_vec.size() << std::endl; // 输出：5

  // 1. 使用std::move将左值big_vec转换为右值引用
  // 此时vector的移动构造函数被调用，直接接管big_vec的内部资源，无拷贝
  std::vector<int> new_vec = std::move(big_vec);

  // 2. 注意：被移动后的原对象（big_vec）处于“有效但未定义”的状态
  // 它仍然可以被析构，也可以重新赋值，但不要访问其原有资源（大小、元素等）
  std::cout << "new_vec 大小：" << new_vec.size() << std::endl; // 输出：5
  std::cout << "big_vec 移动后大小：" << big_vec.size()
            << std::endl; // 输出：0（多数实现下）

  // 3. 对左值变量重复使用std::move
  std::string str = "hello world";
  std::string str1 = std::move(str);  // 移动给str1
  std::string str2 = std::move(str1); // 移动给str2
  std::cout << "str: " << str << std::endl;
  std::cout << "str1: " << str1 << std::endl;
  std::cout << "str2: " << str2 << std::endl; // 输出：hello world

  return 0;
}
