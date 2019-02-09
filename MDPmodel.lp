\* Occupancy solver *\
Minimize
OBJ: __dummy
Subject To
_C1: 0.1 Occupancy_0 = 1
_C2: Occupancy_1 - 0.9 Occupancy_2 = 0
_C3: - 0.9 Occupancy_1 + Occupancy_2 - 0.9 Occupancy_3 = 0
_C4: Occupancy_3 - 0.9 Occupancy_4 = 0
_C5: Occupancy_4 - 0.9 Occupancy_5 = 0
_C6: Occupancy_5 = 0
Bounds
Occupancy_0 <= 9
Occupancy_1 <= 9
Occupancy_2 <= 9
Occupancy_3 <= 9
Occupancy_4 <= 9
Occupancy_5 <= 9
__dummy = 0
End
