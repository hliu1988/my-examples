#include <iostream>
#include <memory> // 包含智能指针头文件

// 自定义测试类
class Test {
public:
  Test(int id) : m_id(id) {
    std::cout << "Test 构造：id = " << m_id << std::endl;
  }
  ~Test() { std::cout << "Test 析构：id = " << m_id << std::endl; }
  void show() const { std::cout << "当前 Test id = " << m_id << std::endl; }

private:
  int m_id;
};

int main() {
  // 1. 创建 unique_ptr，独占管理动态资源
  std::unique_ptr<Test> uptr1(
      new Test(1)); // 直接构造（不推荐，存在异常安全风险）
  std::unique_ptr<Test> uptr2 =
      std::make_unique<Test>(2); // C++14 引入，推荐！更安全、高效

  // 2. 访问所管理的对象（与原始指针用法一致）
  uptr1->show();
  uptr2->show();

  // 3. 移动语义：转移资源所有权（不可直接拷贝）
  // std::unique_ptr<Test> uptr3 = uptr1; // 编译报错：不支持拷贝赋值
  std::unique_ptr<Test> uptr3 =
      std::move(uptr1); // 合法，转移uptr1的资源给uptr3
  if (uptr1 == nullptr) {
    std::cout << "uptr1 资源已被转移，变为空指针" << std::endl;
  }
  uptr3->show(); // 此时uptr3管理原uptr1的资源

  // 4. 手动释放/重置资源
  uptr3.reset(); // 手动释放资源，uptr3变为空指针
  std::unique_ptr<Test> uptr4 = std::make_unique<Test>(4);
  uptr4.reset(new Test(5)); // 释放原有资源，重新管理新资源

  // 5. 获取原始指针（谨慎使用，避免手动修改/释放）
  Test *raw_ptr = uptr2.get();
  raw_ptr->show(); // 可访问，但不要调用 delete raw_ptr;

  return 0; // 函数结束，所有非空unique_ptr自动析构，释放资源
}
