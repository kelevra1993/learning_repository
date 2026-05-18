from typing import List

class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        highest_altitude = 0
        current_altitude = 0

        for delta in gain:
            current_altitude += delta
            highest_altitude = max(current_altitude, highest_altitude)

        return highest_altitude
