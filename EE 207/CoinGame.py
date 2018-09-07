import numpy as np
from matplotlib import pyplot

m = int(input('Enter number of coins per student:'))
n = int(input('Enter number of students:'))
iters = int(input('Enter number of iterations:'))
students = [m] * n
max_coins = 0
for i in range(iters):
    toss = np.random.randint(0, 2)
    s1, s2 = np.random.randint(0, n, 2)
    if toss and students[s2]:
        students[s1] += 1
        students[s2] -= 1
    elif (not toss) and students[s1]:
        students[s1] -= 1
        students[s2] += 1
    max_coins = max(max_coins, students[s1], students[s2])
bins = [0] * (max_coins + 1)
for i in students:
    bins[i] += 1
print(max_coins)
pyplot.hist(students)
pyplot.show()
