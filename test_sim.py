# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:38:43 2018

@author: chong
"""

import set_pump
import get_output
from pyswmm import Simulation
import get_COD_rpt

arg_input_path0 = './sim/arg-new9.inp'
arg_input_path1 = './arg-original.inp'
arg_output_path1 = './arg-original.out'
arg_rpt_path1 = './arg-original.rpt'

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass    

simulation(arg_input_path1)
stand_COD_rain=0
stand_COD_pip=0
for i in range(360):
    data=get_output.read_out(arg_output_path1,i)
    stand_COD_rain+=data['outfall-27'][0][0]
    stand_COD_pip+=data['outfall-5'][0][0]
    stand_COD_pip+=data['outfall-6'][0][0]
print("正常运作的排放")
print("进入污水管的量：",stand_COD_pip)
print("进入自然水体的量：",stand_COD_rain)

print(get_COD_rpt.get_cod(arg_rpt_path1))