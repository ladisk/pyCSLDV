##  Project Roadmap

- [x] Rewrite demodulation functions to take frequency or frequency and damping.
- [x] Add line scan. evaluate_ods, and plot_ods automatically detect 1D or 2D.
- [x] Add mulimodal demodulation. Enable closely spaced modes
- [x] Enable setting polynomial order per mode and per direction
- [x] Support multiple sensors. For example for 3D Scanning LDV. The demodulation function takes real-valued velocity signal of shape
        (n_samples,) or (locations, n_samples)
- [x] Update Showcase.ipynb to explain updated functionalities. Section 5 covers the
      global least-squares fit (several modes, overlapping sidebands, complex poles,
      order per mode and direction, several sensors, line scans) and section 6 the
      rotation of a shape onto other axes
- [x] evaluate_ods, plot_ods and rotate_ods accept the list a multimodal fit
      returns and the stack a multi-sensor one does. plot_ods draws one panel
      per shape
- [x] The design matrix is no longer formed in full: it is built a block of
      rows at a time and only the normal equations are accumulated, so the
      memory no longer follows the record length. The 4527 Hz measurement at
      order 12 went from a 2.7 GB matrix and 13.5 s to 424 MB and 3.7 s, with
      the reconstruction unchanged (MAC 0.9905 against the original suite)
- [x] A fit the data cannot determine is reported: the rank comes back from
      the solve, and the singular values with it, so both a rank-deficient
      system and a merely ill-conditioned one are warned about. Calibrated on
      a plate mode with 5 % noise -- condition 15 still reconstructs at
      MAC 0.999, condition 624 falls to 0.52
- [x] Update testing suite. tests/test_demodulate_1d.py and tests/test_demodulate_2d.py
      cover the least-squares reconstruction; the rotation of a shape onto another
      set of axes is in tests/test_demodulate.py

Note on the two phases, which are easy to confuse. The *scan* phase is the
inertial lag of the mirrors: it is an input (`phi_x`, `phi_y`), measured from
the mirror feedback signals with `reference_phase`, and the reconstruction
needs it to know where the laser was at each sample. The *response* phase is
the arbitrary global phase of the vibration relative to the start of the
acquisition; it is not an input and cannot be, and `align_phase` removes it
afterwards so the real part of the shape can be compared against a real mode
shape. Taking the mirror lags as inputs therefore replaced guessing the scan
phase, not `align_phase`, which is still needed and still exported.