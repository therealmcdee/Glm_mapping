#!/bin/bash


python make_points.py
python generate_field.py testpos.csv
python guess_Glm.py testfield.csv
python compare_G.py
