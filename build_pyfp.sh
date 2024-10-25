#!/bin/bash

python setup.py bdist_wheel
python3 -m pip install --force-reinstall dist/pyfp-0.0.1-cp38-cp38-linux_x86_64.whl