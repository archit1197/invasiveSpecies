\* Occupancy solver *\
Minimize
OBJ: __dummy
Subject To
_C1: 0.372790606398 Occupancy_0 - 0.1062 Occupancy_1 = 1
_C2: - 0.272790606398 Occupancy_0 + 0.4663 Occupancy_1 - 0.1017 Occupancy_2
 = 0
_C3: - 0.2601 Occupancy_1 + 0.4798 Occupancy_2 - 0.0864 Occupancy_3 = 0
_C4: - 0.2781 Occupancy_2 + 0.4609 Occupancy_3 - 0.0999 Occupancy_4 = 0
_C5: - 0.2745 Occupancy_3 + 0.451 Occupancy_4 - 0.6228 Occupancy_5 = 0
_C6: - 0.2511 Occupancy_4 + 0.7228 Occupancy_5 = 0
Bounds
Occupancy_0 <= 9
Occupancy_1 <= 9
Occupancy_2 <= 9
Occupancy_3 <= 9
Occupancy_4 <= 9
Occupancy_5 <= 9
__dummy = 0
End
