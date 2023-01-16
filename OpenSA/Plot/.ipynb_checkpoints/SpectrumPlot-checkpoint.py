import matplotlib.pyplot as plt
def plotspc(data, title):
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label='Spectrum "' + str(i))

    plt.xlabel("Wavelength (nm)")
    plt.ylabel('Intensity')
    plt.title(title)
    plt.show()