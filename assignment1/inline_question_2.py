import numpy as np

l = np.array([[12., 6.],
              [7., 6.]])

test = np.array([[5.], [5.]])

print(np.abs(l - test))

for i in range(l.shape[0]):
    m = np.mean(l[i])
    s = np.std(l[i])

    l[i] = l[i] / s
    test[i] = test[i] / s
    print(s)

print(l)

print(np.abs(l - test))


print(2*np.array([l,l]))
