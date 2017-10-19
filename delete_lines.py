#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/20 下午3:07
# @Author  : LeonHardt
# @File    : train_test_preparation.py

"""
function: Delete the first line of *txt data
Data_from: ./Dendrobium_Standard_Data/*.txt
Save_to: .//Dendrobium_Standard_Data/*.txt
"""

import os
import glob

def delete_first_line(name, start_line=0, end_line=4):
    dir = os.getcwd()
    new_name = dir + '/Dendrobium_Standard_Data/' + name.split('/')[-1]
    if os.path.exists(dir + '/Dendrobium_Standard_Data/') is not True:
        os.makedirs(dir + '/Dendrobium_Standard_Data/')

    f = open(name, 'r', encoding='GBK')
    f_line = f.readlines()
    f_new = open(new_name, 'w')
    new_lines = ''.join(f_line[start_line:end_line])
    f_new.write(new_lines)
    f.close()
    f_new.close()


if __name__ == '__main__':
    name_list = glob.glob(os.getcwd() + '/Dendrobium_Original_Data/' + '*.txt')
    for name_txt in name_list:
        delete_first_line(name_txt, 1, None)
        print('OK')