from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        left_pointer, right_pointer = 0, -1

        length_numbers = len(nums)

        while left_pointer < length_numbers:

            if not nums[left_pointer]:
                if right_pointer == -1:
                    right_pointer = left_pointer

            if nums[left_pointer] and right_pointer != -1:
                nums[right_pointer], nums[left_pointer] = nums[left_pointer], nums[right_pointer]
                right_pointer +=1

            left_pointer += 1

        print(nums)

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums.sort(key=lambda x: x == 0)
        print(nums)

nums = [14,0, 1, 0, 3, 12,0, 1, 0, 3, 12,0, 1, 0, 3, 12]
exercice = Solution()
exercice.moveZeroes(nums=nums)
