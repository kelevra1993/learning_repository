from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def print_linked_list(head):
    node = head
    elements = []

    while node:
        elements.append(str(node.val))
        node = node.next

    print(" -> ".join(elements))


class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:

        fast_pointer = head
        slow_pointer = head
        total_length = 0

        while fast_pointer.next:

            previous_node = slow_pointer
            fast_pointer = fast_pointer.next.next
            slow_pointer = slow_pointer.next
            total_length += 2 if fast_pointer else 1
            if not fast_pointer:
                break

        print(f"Previous Slow Pointer Value : {previous_node.val}")
        print(f"Slow Pointer Value : {slow_pointer.val}")
        print(f"Slow Pointer Next Value : {slow_pointer.next.val if slow_pointer.next else None}")
        print(f"Total Length: {total_length}")

        if total_length:
            previous_node.next = slow_pointer.next
            return head


large_list = list(range(1, 31, 3))
# large_list = [1]
large_list = [1, 2, 3, 4]

nodes = {str(k): ListNode(val=k) for k in large_list}
head = nodes[str(large_list[0])]

length_list = len(large_list)

for index, k in enumerate(large_list):
    next_node = nodes[str(large_list[index + 1])] if index != length_list - 1 else None
    nodes[str(k)].next = next_node

print_linked_list(head)

exercise = Solution()
updated_linked_list = exercise.deleteMiddle(head=head)
print_linked_list(head)
