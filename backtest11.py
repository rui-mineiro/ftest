
import time  # for execution time comparison

import matplotlib.pyplot as plt  # for display purposes

import ruptures as rpt  # our package
from ruptures.metrics import hausdorff

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"



# generate signal
n_samples, dim, sigma = 500, 3, 3
n_bkps = 6  # number of breakpoints
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

print(f"Original Breakpoints:\t{bkps}")

fig, ax_array = rpt.display(signal, bkps)
plt.title("Detected Regimes in S&P 500 Returns")
# plt.show()



algo_python = rpt.Dynp(model="l2", jump=1, min_size=2).fit(signal)  # written in pure python
algo_c = rpt.KernelCPD(kernel="linear", min_size=2).fit(signal)     # written in C



for label, algo in zip(
    ("Python implementation", "C implementation"), (algo_python, algo_c)
):
    start_time = time.time()
    result = algo.predict(n_bkps=n_bkps)
    print(f"{label}:\t{time.time() - start_time:.3f} s")

bkps_python = algo_python.predict(n_bkps=n_bkps)
bkps_c = algo_c.predict(n_bkps=n_bkps)
print(f"Python implementation:\t{bkps_python}")
print(f"C implementation:\t{bkps_c}")
print(f"(Hausdorff distance: {hausdorff(bkps_python, bkps_c):.0f} samples)")

# number of changes to detect is unknown 


algo_python = rpt.Pelt(model="l2", jump=1, min_size=10).fit(
    signal
)  # written in pure python
algo_c = rpt.KernelCPD(kernel="linear", min_size=2).fit(
    signal
)  # written in C, same class as before


penalty_value = 100  # beta

for label, algo in zip(
    ("Python implementation", "C implementation"), (algo_python, algo_c)
):
    start_time = time.time()
    result = algo.predict(pen=penalty_value)
    print(f"{label}:\t{time.time() - start_time:.3f} s")

    bkps_python = algo_python.predict(pen=penalty_value)
bkps_c = algo_c.predict(pen=penalty_value)
print(f"Python implementation:\t{bkps_python}")
print(f"C implementation:\t{bkps_c}")
print(f"(Hausdorff distance: {hausdorff(bkps_python, bkps_c):.0f} samples)")

params = {"gamma": 1e-2}
algo_python = rpt.Dynp(model="rbf", params=params, jump=1, min_size=2).fit(
    signal
)  # written in pure python
algo_c = rpt.KernelCPD(kernel="rbf", params=params, min_size=2).fit(
    signal
)  # written in C

for label, algo in zip(
    ("Python implementation", "C implementation"), (algo_python, algo_c)
):
    start_time = time.time()
    result = algo.predict(n_bkps=n_bkps)
    print(f"{label}:\t{time.time() - start_time:.3f} s")

bkps_python = algo_python.predict(n_bkps=n_bkps)
bkps_c = algo_c.predict(n_bkps=n_bkps)
print(f"Python implementation:\t{bkps_python}")
print(f"C implementation:\t{bkps_c}")
print(f"(Hausdorff distance: {hausdorff(bkps_python, bkps_c):.0f} samples)")


algo_python = rpt.Pelt(model="rbf", jump=1, min_size=2).fit(
    signal
)  # written in pure python
algo_c = rpt.KernelCPD(kernel="rbf", min_size=2).fit(
    signal
)  # written in C, same class as before


penalty_value = 1  # beta

for label, algo in zip(
    ("Python implementation", "C implementation"), (algo_python, algo_c)
):
    start_time = time.time()
    result = algo.predict(pen=penalty_value)
    print(f"{label}:\t{time.time() - start_time:.3f} s")

bkps_python = algo_python.predict(pen=penalty_value)
bkps_c = algo_c.predict(pen=penalty_value)
print(f"Python implementation:\t{bkps_python}")
print(f"C implementation:\t{bkps_c}")
print(f"(Hausdorff distance: {hausdorff(bkps_python, bkps_c):.0f} samples)")



