## A Non-Dimensional Scaling Approach to Modeling Material Pyrolysis

This repository contains the data, scripts, and technical documents required to generate the pre-print for this paper submitted to the 2024 Combustion Symposium.

## Instructions
* Clone this repository with submodules
  ```
  git clone --recurse-submodules https://github.com/johodges/2024combustion_institute
  ```
* Pre-process data
  ```
  cd 2024combustion_institute/scripts
  python fsri_collect_thermophysical_properties.py
  python fsri_collect_cone_data.py
  python process_fsri_database.py
  python process_faa_data.py
  python proces_fpl_data.py
  ```
