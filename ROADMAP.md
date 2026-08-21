##  Project Roadmap

- [x] Rewrite demodulation functions to take frequency or frequency and damping.
- [x] Add line scan. evaluate_ods, and plot_ods automatically detect 1D or 2D.
- [x] Add mulimodal demodulation. Enable closely spaced modes
- [x] Enable setting polynomial order per mode and per direction
- [x] Support multiple sensors. For example for 3D Scanning LDV. The demodulation function takes real-valued velocity signal of shape
        (n_samples,) or (locations, n_samples)
- [ ] Update Showcase.ipynb to explain updated functionalities
- [ ] Update testing suite

Note: The new implementation takes the mirror lags as an inputs. This replaces the align_phase function