import numpy as np
import time

import numpy as np
import matplotlib.pyplot as plt
from FA import FA
import math

fa = FA(2, 40, 1, 0.000001, 0.97, 50, [-100, 100], 3)
list = np.argsort(fa.FitnessValue)

print(fa.I_average_Distance(3,list[0]))