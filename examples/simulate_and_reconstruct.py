"""
Virtual CSLDV experiment: simulate the LDV velocity signal measured along a
Lissajous scan over a vibrating plate, reconstruct the operating deflection
shape with the sideband-demodulation approach and compare it to the exact
shape using the MAC.

Run from the project base directory with:

    python -m examples.simulate_and_reconstruct
"""

import matplotlib.pyplot as plt

import pycsldv

# Measurement parameters
fs = 25000.0        # sampling frequency [Hz]
duration = 10.0     # measurement duration [s]
fx = 1.4            # x scan frequency [Hz]
fy = 8.0            # y scan frequency [Hz]
fz = 2000.0         # response frequency [Hz]
order = 8           # Chebyshev order of the reconstruction

n_samples = int(duration * fs)
print(f"Lissajous closure period: {pycsldv.scan_period(fx, fy):.1f} s")

# Simulate the measurement (plate mode (2, 3), 5 % noise)
shape = pycsldv.plate_mode(2, 3)
t, x, y = pycsldv.lissajous(fx, fy, n_samples, fs)
velocity = pycsldv.simulate_response(shape, fz, x, y, fs, noise_std=0.05)

# Reconstruct the ODS
coefficients = pycsldv.demodulate_ods(velocity, x, y, fs, fx, fy, fz, order=order)
coefficients = pycsldv.align_phase(coefficients)

# Compare with the exact shape
x_grid, y_grid, z = pycsldv.evaluate_ods(coefficients, resolution=100)
mac = pycsldv.mac(shape(x_grid, y_grid), z.real)
print(f"MAC(exact, reconstructed) = {mac:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pycsldv.plot_trajectory(x[: int(fs * pycsldv.scan_period(fx, fy))],
                        y[: int(fs * pycsldv.scan_period(fx, fy))], ax=axes[0])
axes[0].set_title("Scan trajectory (one period)")
pycsldv.plot_ods(coefficients, ax=axes[1])
axes[1].set_title(f"Reconstructed ODS at {fz:.0f} Hz")
fig.tight_layout()
plt.show()
