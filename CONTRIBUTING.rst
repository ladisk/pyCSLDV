Contributing to pyCSLDV
-----------------------

Contributions are welcome! To contribute:

1. Fork the repository and clone your fork.
2. Create a virtual environment and install the development requirements:

   .. code-block:: console

       $ uv venv
       $ uv pip install -e ".[dev]"

3. Create a feature branch, make your changes and add unit tests in
   ``tests/`` (the test suite uses `pytest <https://docs.pytest.org>`_).
4. Run the tests locally:

   .. code-block:: console

       $ pytest

5. Open a pull request with a clear description of the change.

Please keep the code style consistent with the existing modules
(NumPy-style docstrings, descriptive parameter names) and document any new
public function in ``docs/source/code.rst``.

When changing the package version, use ``sync_version.py`` so that
``pyproject.toml``, ``pycsldv/__init__.py`` and ``docs/source/conf.py``
stay consistent:

.. code-block:: console

    $ python sync_version.py --bump patch
