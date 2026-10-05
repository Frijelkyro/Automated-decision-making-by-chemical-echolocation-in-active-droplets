import re
import numpy as np
import matplotlib.pyplot as plt

with open("data/param.txt.bak") as f:
    s = f.read()

def get_array(name):
    m = re.search(rf"{name}:\s*\[(.*?)\]", s, re.S)
    return np.fromstring(m.group(1), sep=" ")

birth = get_array("birth_times")
exit_time = get_array("exit_trigger_time")

delay = exit_time - birth
mask = np.isfinite(delay)

plt.figure(figsize=(9, 4))
plt.plot(birth[mask], delay[mask], ".", ms=4)
plt.xlabel("Birth time")
plt.ylabel("Exit trigger time − birth time")
plt.title("Particle exit delay")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()