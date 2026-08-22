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
- [ ] The design matrix is n_samples x 2(Px+1)(Py+1) and is formed in full: a
      10 s record at 100 kS/s and order 12 needs 2.7 GB, 30 s needs 8.1 GB.
      Accumulating the normal equations in chunks would make the memory
      independent of the record length (0.9 MB at order 12), at the cost of
      squaring the condition number
- [x] A fit the data cannot determine is reported: the rank comes back from
      the solve, and the singular values with it, so both a rank-deficient
      system and a merely ill-conditioned one are warned about. Calibrated on
      a plate mode with 5 % noise -- condition 15 still reconstructs at
      MAC 0.999, condition 624 falls to 0.52
- [x] Update testing suite. tests/test_demodulate_1d.py and tests/test_demodulate_2d.py
      cover the least-squares reconstruction; the rotation of a shape onto another
      set of axes is in tests/test_demodulate.py

Note: The new implementation takes the mirror lags as an inputs. This replaces the align_phase function