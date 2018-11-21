# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:47:58 2018

@author: chong
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:23:08 2018

@author: chong
"""

from yaml import load

constants = load(open('./constants.yml', 'r', encoding='utf-8'))

def handle_line(line, flag, title):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    return flag

def get_cod(filename):
    with open(filename, 'rt') as data:
        pumps_flag =  False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('External Outflow')>=0:
                    data=node[3]

    return data
        

if __name__ == '__main__':

    filename='./arg-original.rpt'
    #arg_output_path0 = './sim/arg-original.rpt'
    print(get_cod(filename))

