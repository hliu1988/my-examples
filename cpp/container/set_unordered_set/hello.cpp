#include <iostream>
#include <set>
#include <unordered_set>
using namespace std;

int main() {
  // 1. set：有序存储
  set<int> s1 = {3, 1, 2, 5, 4};
  cout << "set遍历（有序）：";
  for (int num : s1) {
    cout << num << " "; // 输出：1 2 3 4 5
  }
  cout << endl;

  // 2. unordered_set：无序存储
  unordered_set<int> s2 = {3, 1, 2, 5, 4};
  cout << "unordered_set遍历（无序）：";
  for (int num : s2) {
    cout << num << " "; // 输出（示例）：3 1 2 5 4（顺序不固定）
  }
  cout << endl;

  return 0;
}
