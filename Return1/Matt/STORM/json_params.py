#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:04:54 2020

@author: mattarnold
"""

import json

parameters = {
        "file name:" : "/Users/mattarnold/Masters/Return1/Matt/STORM/cameraman.tif",
        "filter parameters:" : {
                "filter type:" : "difference of Gaussians",
                "input parameter a" : 40,
                "input parameter b" : 40,
        }
}

with open("DOG_params.json", "w") as write_file:
    json.dump(parameters, write_file)