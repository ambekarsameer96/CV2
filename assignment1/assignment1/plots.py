import numpy as np
import matplotlib.pyplot as plt
import pickle

def RMS_it_plot(pickle_data, title):
    with open(pickle_data, 'rb') as f:
        RMS = pickle.load(f)

    for key, value in RMS.items():
        plt.plot(value, label=key)

    plt.legend()
    plt.show()

RMS_it_plot("unif_RMS_li.pl", title="RMS/it for uniform sampling with different ratios")
