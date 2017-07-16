import sys
import matplotlib.pyplot as plt

plt.style.use('ggplot')

with open(sys.argv[1], 'r') as f:
    losses = [float(l) for l in f]

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('time')
plt.show()