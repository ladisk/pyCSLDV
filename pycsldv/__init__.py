"""
pyCSLDV — continuous scanning laser Doppler vibrometry.

Scan-trajectory design, virtual experiments and operating-deflection-shape
reconstruction using the polynomial (sideband demodulation) approach.
"""

__version__ = "0.1.0"

from .scan import (lissajous, sinusoidal_scan, scan_period, drive_signals,
                   normalize_scan, dominant_frequency)
from .simulate import simulate_response, plate_mode, plate_frequency, chebyshev_shape
from .demodulate import demodulate_ods, demodulate_ods_1d, demodulate_ods_2d, evaluate_ods, align_phase, reference_phase
from .mac import mac
from .visualize import plot_trajectory, plot_ods
