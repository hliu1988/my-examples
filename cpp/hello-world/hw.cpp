#include <iostream>

int main(int argc, char **argv) {
  std::cout << "argc = " << argc << std::endl;
  auto *name = argv[1];
  std::cout << "Hello World:" << std::endl << name << std::endl;
  return 0;
}
