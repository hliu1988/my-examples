#include <cstring>
#include <iostream>
#include <utility> // 包含std::move

class MyString {
public:
  // 1. 默认构造函数
  MyString() : m_data(nullptr), m_size(0) {
    std::cout << "默认构造函数被调用" << std::endl;
  }

  // 2. 普通构造函数（创建动态数组）
  MyString(const char *str) {
    std::cout << "普通构造函数被调用" << std::endl;
    if (str == nullptr) {
      m_data = nullptr;
      m_size = 0;
      return;
    }
    m_size = strlen(str);
    // 分配内存，拷贝数据
    m_data = new char[m_size + 1];
    strcpy(m_data, str);
  }

  // 3. 拷贝构造函数（处理左值，深拷贝）
  MyString(const MyString &other) {
    std::cout << "拷贝构造函数被调用" << std::endl;
    if (other.m_data == nullptr) {
      m_data = nullptr;
      m_size = 0;
      return;
    }
    // 步骤1：为新对象分配新内存（深拷贝核心）
    m_size = other.m_size;
    m_data = new char[m_size + 1];
    // 步骤2：拷贝源对象的全部数据
    strcpy(m_data, other.m_data);
  }

  // 4. 移动构造函数（处理右值，资源转移）
  MyString(MyString &&other) noexcept {
    std::cout << "移动构造函数被调用" << std::endl;
    // 步骤1：接管源对象的资源（仅修改指针指向，无内存分配）
    m_data = other.m_data;
    m_size = other.m_size;

    // 步骤2：将源对象的资源置空，避免析构时重复释放
    other.m_data = nullptr;
    other.m_size = 0;
  }

  // 5. 析构函数
  ~MyString() {
    std::cout << "析构函数被调用，释放内存：" << (m_data ? m_data : "nullptr")
              << std::endl;
    if (m_data != nullptr) {
      delete[] m_data;
      m_data = nullptr;
    }
  }

  // 辅助方法：打印字符串内容
  void print() const {
    if (m_data != nullptr) {
      std::cout << "字符串内容：" << m_data << "，大小：" << m_size
                << std::endl;
    } else {
      std::cout << "字符串为空，大小：" << m_size << std::endl;
    }
  }

private:
  char *m_data;  // 指向动态分配的字符数组
  size_t m_size; // 字符串大小（不含结束符）
};

int main() {
  // 场景1：用临时对象（右值）初始化新对象 → 触发移动构造函数
  std::cout << "===== 场景1：临时对象初始化 =====" << std::endl;
  MyString str1 = MyString("hello world"); // 临时对象是右值
  str1.print();

  // 场景2：用std::move转换左值 → 触发移动构造函数
  std::cout << "\n===== 场景2：std::move转换左值 =====" << std::endl;
  MyString str2("test move");      // str2是左值
  MyString str3 = std::move(str2); // 转换为右值，触发移动构造
  str3.print();
  str2.print(); // 原对象str2处于“有效但未定义”状态（资源已被转移）

  // 场景3：用左值初始化新对象 → 触发拷贝构造函数
  std::cout << "\n===== 场景3：左值初始化 =====" << std::endl;
  MyString str4("test copy");
  MyString str5 = str4; // 左值初始化，触发拷贝构造
  str5.print();
  str4.print(); // 原对象str4保持不变

  return 0;
}
