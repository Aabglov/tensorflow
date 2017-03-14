import math, random
from fractions import gcd
import numpy as np

pi_ests = []
for j in range(1000):
    # Simulate D20 rolls
    MIN = 1
    MAX = 100
    NUM_ROLLS = 200

    coprime = 0
    cofactor = 0

    for _ in range(NUM_ROLLS):
        # Generate 2 random values in our range
        dice_1 = random.randint(MIN,MAX)
        dice_2 = random.randint(MIN,MAX)

        # Determine if they're coprime or not
        g = gcd(dice_1,dice_2)
        if g==1:
            coprime += 1
        elif g>1:
            cofactor += 1

    x = coprime / NUM_ROLLS
    pi_est = math.sqrt(6.0 / x)
    pi_ests.append(pi_est)
    #print("Estimate of PI: {}".format(pi_est))


print("Average estimate of pi: {}".format(np.mean(pi_ests)))
