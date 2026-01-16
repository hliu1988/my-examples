// 来自 CoreMark 的链表操作逻辑
typedef struct list_node {
  struct list_node *next;
  int data;
} list_node;

int list_search(list_node *head, int target) {
  list_node *curr = head;
  if (curr == nullptr) {
    goto not_found; // 跳转A
  }
loop:
  if (curr->data == target) {
    goto found; // 跳转B
  }
  curr = curr->next;
  if (curr != nullptr) {
    goto loop; // 跳转C（循环跳转）
  }
not_found:
  return 0; // 最终结果1
found:
  return 1; // 最终结果2
}
