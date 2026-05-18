class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        merged = ""

        longest_word = word1 if len(word1) > len(word2) else word2

        length_word_1 = len(word1)
        length_word_2 = len(word2)

        maximum_length = max(length_word_1, length_word_2)
        minimum_length = min(length_word_1, length_word_2)

        for i in range(maximum_length):

            merged += word1[i] + word2[i]
            print(merged)
            if i == minimum_length-1:
                if maximum_length != minimum_length:
                    merged += longest_word[i+1:]
                return merged


word1 = "ab"
word2 = "pqrs"
exercice = Solution()
print(exercice.mergeAlternately(word1=word1, word2=word2))
