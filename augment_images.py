#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:56:46 2018

@author: dy
"""

import Augmentor
p = Augmentor.Pipeline('/home/dy/dy/smart_design/sketch_2')
#p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.0, max_factor=1.5)
p.sample(100)