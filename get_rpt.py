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

def get_rpt(filename):
    with open(filename, 'rt') as data:
        total_in=0
        flooding=0
        store=0
        outflow=0
        upflow=0
        downflow=0
        pumps_flag = outfall_flag= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('External Outflow')>=0 or\
                   line.find('Exfiltration Loss')>=0 or \
                   line.find('Mass Reacted')>=0:
                    outflow+=float(node[3])

                elif line.find('Flooding Loss')>=0:
                    flooding=float(node[3])
                elif line.find('Final Stored Mass')>=0:
                    store=float(node[4])
                elif line.find('Dry Weather Inflow')>=0 or \
                     line.find('Wet Weather Inflow')>=0 :
                    total_in+=float(node[4])
                    
                elif line.find('Groundwater Inflow')>=0 or \
                     line.find('RDII Inflow')>=0 or \
                     line.find('External Inflow')>=0:
                    total_in+=float(node[3])
                    
            if outfall_flag and node!=[]:
                if line.find('outfall-27')>=0 or line.find('outfall-28')>=0 or line.find('outfall-24')>=0:
                    upflow+=float(node[5])
                elif line.find('outfall-')>=0 or line.find('12')>=0 or line.find('67')>=0:
                    downflow+=float(node[5])
                    

    return total_in,flooding,store,outflow,upflow,downflow
        

if __name__ == '__main__':

    filename='./ot.rpt'
    #arg_output_path0 = './sim/arg-original.rpt'
    print(get_rpt(filename))

