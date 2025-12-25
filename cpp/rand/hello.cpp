#include <cstdlib>
#include <iostream>
using namespace std;

int main() {
  // 代表标准库函数rand()能够生成的最大整数值
  cout << "RAND_MAX = " << RAND_MAX << endl;

  // 用系统时间初始化随机数种子
  srand((unsigned int)time(NULL));

  cout << "rand_val = " << rand() / (float)RAND_MAX << endl;
  cout << "rand_val = " << rand() / (float)RAND_MAX << endl;
  return 0;
}
