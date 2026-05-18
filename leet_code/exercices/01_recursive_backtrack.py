"""
Given a list(range(n)), generate all the subset combinations.
For instance given the list [1, 2, 3]
the solution will be [ [], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3] ]
"""
number_elements = 3
input_list = list(range(1, number_elements + 1))

final_results = []
buffer = []


def back_track(index):
    if index == number_elements + 1:
        final_results.append(buffer[:])
        return

    # if left don't take anything just keep going
    back_track(index + 1)

    # if right do something else
    buffer.append(index)
    back_track(index + 1)
    buffer.pop()

    return


# back_track(1)
# print(final_results, sep=" | ")

from collections import deque

personal_solution = []

stack = deque()
stack.append([])
index = 1

while len(stack) < 2 ** number_elements:

    left_most_element = stack.popleft()

    stack.append(left_most_element)
    stack.append(left_most_element + [index])

    if len(stack) == 2 ** index:
        index += 1
    # print(list(stack))

print(list(stack))
