from typing import List


class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        num_1_set = set(nums1)
        num_2_set = set(nums2)

        return [list(num_1_set - num_2_set), list(num_2_set - num_1_set)]


nums1 = [1, 2, 3, 4, 5, 6]
nums2 = [2, 4, 6]

exercice = Solution()
result = exercice.findDifference(nums1=nums1, nums2=nums2)
print(result)
