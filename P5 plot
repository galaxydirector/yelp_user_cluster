import matplotlib.pyplot as plt

k = [5, 10, 20, 30, 50, 100, 200, 300, 500]
mean = [0.41, 0.33, 0.35, 0.36, 0.43, 0.451, 0.52, 0.63, 0.75]
minimum = [0.49, 0.2, 0.37, 0.44, 0.56, 0.51, 0.62, 0.63, 0.85]
maximum = [0.31, 0.13, 0.25, 0.26, 0.43, 0.451, 0.562, 0.613, 0.95]

plt.title('Error based on the initialization')
plt.plot(k, mean, color='green', label='mean distance')
plt.plot(k, minimum, color='red', label='minimum distance')
plt.plot(k, maximum, color='skyblue', label='maximum distance')
plt.legend()

plt.xlabel('k value')
plt.ylabel('distance')
plt.show()
