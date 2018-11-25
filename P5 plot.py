import matplotlib.pyplot as plt


def error_plot(k, mean, minimum, maximum):
    plt.title('Error based on the initialization')
    plt.plot(k, mean, color='green', label='mean distance')
    plt.plot(k, minimum, color='red', label='minimum distance')
    plt.plot(k, maximum, color='blue', label='maximum distance')
    plt.legend()
    plt.xlabel('k value')
    plt.ylabel('distance')
    plt.show()


error_plot(k, mean_distance, min_distance, max_distance)
