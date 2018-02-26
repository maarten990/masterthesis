import sys
import matplotlib.pyplot as plt

plt.style.use('ggplot')

for path in sys.argv[1:]:
    with open(path, 'r') as f:
        losses = [float(l) for l in f]

    plt.plot(losses, label=path)

plt.legend()
plt.ylabel('loss')
plt.xlabel('time')
plt.show()
