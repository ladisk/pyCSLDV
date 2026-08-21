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
- [ ] evaluate_ods and plot_ods take a single coefficient array, so the list a
      multimodal demodulation returns has to be indexed by the caller
- [ ] rotate_ods takes a single 2D matrix, not the stacked multi-sensor or
      multimodal results
- [ ] The design matrix is n_samples x 2(Px+1)(Py+1) and is formed in full: a
      10 s record at 100 kS/s and order 12 needs 2.7 GB
- [ ] A record shorter than one scan period is fitted without warning, and the
      result is meaningless
- [x] Update testing suite. tests/test_demodulate_1d.py and tests/test_demodulate_2d.py
      cover the least-squares reconstruction; the rotation of a shape onto another
      set of axes is in tests/test_demodulate.py

Note: The new implementation takes the mirror lags as an inputs. This replaces the align_phase function