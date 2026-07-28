"""
pyCSLDV — continuous scanning laser Doppler vibrometry.

Scan-trajectory design, virtual experiments and operating-deflection-shape
reconstruction using the polynomial (sideband demodulation) approach.
"""

__version__ = "0.1.0"

from .scan import lissajous, scan_period, drive_signals
from .simulate import simulate_response, plate_mode, chebyshev_shape
from .demodulate import demodulate_ods, evaluate_ods, align_phase, reference_phase
from .mac import mac
from .visualize import plot_trajectory, plot_ods
