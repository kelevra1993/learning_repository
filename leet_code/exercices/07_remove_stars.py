from typing import List


class Solution:
    def removeStars(self, s: str) -> str:
        stack = []

        for element in s:
            if element == "*":
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(element)

        return "".join(stack)


s = "leet**cod*e"
s = "*éertdg****erase*****"

exercice = Solution()
result = exercice.removeStars(s=s)
print(result)
