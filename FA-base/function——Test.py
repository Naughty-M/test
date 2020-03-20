import math

import numpy as np
D = 3
list2 = [1,1,1]
list2=np.array(list2)
print(list2)
list1 = [2,2,3]
print(list1*list2)
print(np.sqrt(list1*list2))

print((2*np.sqrt(np.random.rand(D))-1)*(np.random.rand(D))/np.random.rand(D))
