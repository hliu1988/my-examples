#include <iostream>

using namespace std;

typedef struct {
  float a;
  char b;
} st1_t;

struct st2_t {
  char b;
  float a;
  char c;
};

int main() {
  st1_t t1;
  struct st2_t t2;
  cout << sizeof(t1) << "\n" << sizeof(t2) << endl;
}
