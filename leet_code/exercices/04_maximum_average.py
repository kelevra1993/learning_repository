from typing import List
import numpy as np


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        largest_mean = np.mean(nums[:k])
        current_mean = np.mean(nums[:k])

        for i in range(1, len(nums) - k + 1):

            left = nums[i-1] / k
            right = nums[i + k-1] / k

            print(left)
            print(right)

            current_mean = current_mean + right - left
            largest_mean = current_mean if current_mean > largest_mean else largest_mean

        return largest_mean


nums = [0, 1, 1, 3, 3]
k = 4

exercice = Solution()
result = exercice.findMaxAverage(nums=nums, k=k)
print(result)
