import random
import math


def sphere(x):
    ans = 0
    for i in range(len(x)):
        ans += x[i] ** 2
    return ans


# Employee Bee
def EBee(X, f, trials):
    for i in range(len(X)):
        V = []
        R = X.copy()
        R.remove(X[i])
        r = random.choice(R)

        for j in range(len(X[0])):
            V.append((X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j])))

        if f(X[i]) < f(V):
            trials[i] += 1
        else:
            X[i] = V
            trials[i] = 0
    return X, trials


def P(X, f):
    P = []
    sP = sum([1 / (1 + f(i)) for i in X])
    for i in range(len(X)):
        P.append((1 / (1 + f(X[i]))) / sP)

    return P


# Onlooker Bee
def OBee(X, f, trials):
    Pi = P(X, f)

    for i in range(len(X)):
        if random.random() < Pi[i]:
            V = []
            R = X.copy()
            R.remove(X[i])
            r = random.choice(R)

            for j in range(len(X[0])):  # x[0] or number of dimensions
                V.append((X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j])))

            if f(X[i]) < f(V):
                trials[i] += 1
            else:
                X[i] = V
                trials[i] = 0
    return X, trials


# Scout Bee
def SBee(X, trials, bounds, limit=3):
    for i in range(len(X)):
        if trials[i] > limit:
            trials[i] = 0
            X[i] = [bounds[i][0] + (random.uniform(0, 1) * (bounds[i][1] - bounds[i][0])) for i in range(len(X[0]))]
    return X
