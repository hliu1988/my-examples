#include <iostream>
#include <memory>

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
  // 1. 创建 shared_ptr，共享管理动态资源
  std::shared_ptr<Test> sptr1(new Test(1)); // 直接构造（不推荐）
  std::shared_ptr<Test> sptr2 = std::make_shared<Test>(2); // 推荐！更高效、安全

  // 2. 查看引用计数
  std::cout << "sptr1 引用计数：" << sptr1.use_count() << std::endl; // 输出：1
  std::shared_ptr<Test> sptr3 = sptr1; // 拷贝，引用计数+1
  std::cout << "sptr1 引用计数：" << sptr1.use_count() << std::endl; // 输出：2
  std::cout << "sptr3 引用计数：" << sptr3.use_count() << std::endl; // 输出：2

  // 3. 共享资源访问
  sptr3->show();
  sptr1->show(); // 多个shared_ptr访问同一个资源

  // 4. 移动语义与引用计数变化
  std::shared_ptr<Test> sptr4 =
      std::move(sptr2); // 移动，sptr2的资源转移给sptr4，引用计数不变（仍为1）
  if (sptr2 == nullptr) {
    std::cout << "sptr2 资源已被转移，变为空指针" << std::endl;
  }
  std::cout << "sptr4 引用计数：" << sptr4.use_count() << std::endl; // 输出：1

  // 5. 重置资源，引用计数变化
  sptr3.reset(); // sptr3释放资源，引用计数-1
  std::cout << "sptr1 引用计数（sptr3重置后）：" << sptr1.use_count()
            << std::endl; // 输出：1

  return 0; // 函数结束，剩余shared_ptr销毁，引用计数归0，资源被释放
}
