import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def create_sine(tuple):
    samples = np.arange(1,1001) / 1000
    return tuple[0]*np.sin(2 * np.pi * tuple[1]*samples + tuple[2])
def f(x):
    return x*2-1
def addSines(rangesum):
    arrOfSines = []
    for i in rangesum:
        arrOfSines.append(create_sine((1/f(i),f(i),0)))
    return sum(arrOfSines)
sines = []
data = [(1,1,0),(0.4,3,0),(1.7,1,np.pi/2)] # (amp,freq,phase)
for item in data:
    sines.append(create_sine(item))


fig, ax = plt.subplots()
ax.scatter(np.arange(1,1001), sines[0])
ax.scatter(np.arange(1,1001), sines[1])
ax.scatter(np.arange(1,1001), sines[2])
plt.show()

fig2, ax2 = plt.subplots()
ax2.scatter(np.arange(1,1001), create_sine((1,1,0)) + create_sine((1/3,3,0)))
plt.show()

#range = np.arange(1,2001)
#sum = addSines(range)

#fig3, ax3 = plt.subplots()
#ax3.scatter(np.arange(1,1001), sum)
#plt.show()
#print('end')

rangeNew = np.arange(1,6)
sumOfFive = addSines(rangeNew)

fig4, ax4 = plt.subplots()
ax4.scatter(np.arange(1,1001), sumOfFive)
plt.show()

yf = fft(sumOfFive)
xf = fftfreq(1000, 1 / 1000)

plt.plot(xf, np.abs(yf))
plt.show()
print('end')

