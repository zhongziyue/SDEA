import random

import torch as t
import numpy as np

seed = 11037
random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)
np.random.seed(seed)