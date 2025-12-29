#include <iostream>
#include <memory>

// 前向声明
class B;

class A {
public:
  // 用 weak_ptr 引用 B，避免循环引用
  std::weak_ptr<B> m_b_ptr;
  A() { std::cout << "A 构造" << std::endl; }
  ~A() { std::cout << "A 析构" << std::endl; }
  void showB();
};

class B {
public:
  // 用 weak_ptr 引用 A，避免循环引用
  std::weak_ptr<A> m_a_ptr;
  B() { std::cout << "B 构造" << std::endl; }
  ~B() { std::cout << "B 析构" << std::endl; }
  void showA();
};

void A::showB() {
  // lock() 转换为 shared_ptr，若资源未释放则返回有效shared_ptr，否则返回空
  std::shared_ptr<B> b_ptr = m_b_ptr.lock();
  if (b_ptr) {
    std::cout << "A 成功访问 B" << std::endl;
  } else {
    std::cout << "B 资源已被释放" << std::endl;
  }
}

void B::showA() {
  std::shared_ptr<A> a_ptr = m_a_ptr.lock();
  if (a_ptr) {
    std::cout << "B 成功访问 A" << std::endl;
  } else {
    std::cout << "A 资源已被释放" << std::endl;
  }
}

int main() {
  {
    std::shared_ptr<A> a_ptr = std::make_shared<A>();
    std::shared_ptr<B> b_ptr = std::make_shared<B>();

    // 建立相互引用（若用 shared_ptr 则会形成循环引用，导致资源无法释放）
    a_ptr->m_b_ptr = b_ptr;
    b_ptr->m_a_ptr = a_ptr;

    a_ptr->showB();
    b_ptr->showA();

    // 查看引用计数（仍为1，因为 weak_ptr 不增加计数）
    std::cout << "a_ptr 引用计数：" << a_ptr.use_count() << std::endl;
    std::cout << "b_ptr 引用计数：" << b_ptr.use_count() << std::endl;
  } // 作用域结束，a_ptr、b_ptr 销毁，引用计数归0，A、B 正常析构（无内存泄漏）

  return 0;
}
