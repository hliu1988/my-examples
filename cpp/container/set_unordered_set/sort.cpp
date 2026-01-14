#include <chrono> // 计时
#include <iostream>
#include <set>
#include <unordered_set>
using namespace std;

int main() {
  const int N = 100000; // 测试数据量
  set<int> s;
  unordered_set<int> us;

  // 测试插入效率
  auto start = chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    s.insert(i);
  }
  auto end = chrono::high_resolution_clock::now();
  cout << "set插入" << N << "个元素耗时："
       << chrono::duration_cast<chrono::microseconds>(end - start).count()
       << "μs" << endl;

  start = chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    us.insert(i);
  }
  end = chrono::high_resolution_clock::now();
  cout << "unordered_set插入" << N << "个元素耗时："
       << chrono::duration_cast<chrono::microseconds>(end - start).count()
       << "μs" << endl;

  // 测试查找效率
  start = chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    s.find(i);
  }
  end = chrono::high_resolution_clock::now();
  cout << "set查找" << N << "个元素耗时："
       << chrono::duration_cast<chrono::microseconds>(end - start).count()
       << "μs" << endl;

  start = chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    us.find(i);
  }
  end = chrono::high_resolution_clock::now();
  cout << "unordered_set查找" << N << "个元素耗时："
       << chrono::duration_cast<chrono::microseconds>(end - start).count()
       << "μs" << endl;

  return 0;
}
