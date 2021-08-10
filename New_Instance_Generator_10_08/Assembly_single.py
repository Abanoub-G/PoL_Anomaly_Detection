#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 20:24:15 2021

@author: dentalcare999
"""
#
#  elif(pattern_stratigies_list[i] == 'awgn_signle'):
#   temp = np.load('single_false_augmentation_1.npy',allow_pickle=True).item()      
#   pattern_dict['awgn_signle'].append(temp)
#
#  elif(pattern_stratigies_list[i] == 'motion_signle'):
#   temp = np.load('single_false_augmentation_2.npy',allow_pickle=True).item()      
#   pattern_dict['motion_signle'].append(temp) 
#
#  elif(pattern_stratigies_list[i] == 'reduced_signle'):
#   temp = np.load('single_false_augmentation_3.npy',allow_pickle=True).item()       
#   pattern_dict['reduced_signle'].append(temp)


import numpy as np

#pattern_stratigies_list = ['awgn','motion','reduced']
#
#classes_in_each_pattern_dict ={}
#classes_in_each_pattern_dict['awgn'] =[0,1]
#classes_in_each_pattern_dict['motion'] =[2,3]
#classes_in_each_pattern_dict['reduced'] =[3,4]
#
#size_of_class_pattern_dict = {}
#size_of_class_pattern_dict['awgn'] ={0:10,1:10}
#size_of_class_pattern_dict['motion'] = {2:10,3:10}
#size_of_class_pattern_dict['reduced'] = {3:10,4:10}
#
#accuracy_each_class_pattern_dict = {}
#accuracy_each_class_pattern_dict['awgn'] ={0:0.5,1:0.5}
#accuracy_each_class_pattern_dict['motion'] = {2:0.5,3:0.5}
#accuracy_each_class_pattern_dict['reduced'] = {3:0.5,4:0.5}

pattern_stratigies_list = ['awgn','motion','reduced']

classes_in_each_pattern_dict ={}
classes_in_each_pattern_dict['awgn'] =[4]
classes_in_each_pattern_dict['motion'] =[9]
classes_in_each_pattern_dict['reduced'] =[1]

size_of_each_class_in_each_pattern_dict = {}
size_of_each_class_in_each_pattern_dict['awgn'] ={4:20}
size_of_each_class_in_each_pattern_dict['motion'] ={9:20}
size_of_each_class_in_each_pattern_dict['reduced'] ={1:20}

accuracy_of_each_class_in_each_pattern_dict = {}
accuracy_of_each_class_in_each_pattern_dict['awgn'] ={4:0.5}
accuracy_of_each_class_in_each_pattern_dict['motion'] = {9:0.5}
accuracy_of_each_class_in_each_pattern_dict['reduced'] ={1:0.5}

def single_set_generator(pattern_stratigies_list,classes_in_each_pattern_dict,size_of_each_class_in_each_pattern_dict,accuracy_of_each_class_in_each_pattern_dict):
 pattern_dict = {}   
 for i in range(0,len(pattern_stratigies_list)): 
  if(pattern_stratigies_list[i] == 'awgn'):
   temp = np.load('sampling_with_selector_awgn.npy',allow_pickle=True).item()   
   pattern_dict['awgn'] = temp
   
  elif(pattern_stratigies_list[i] == 'motion'):
   temp = np.load('sampling_with_selector_motion.npy',allow_pickle=True).item()     
   pattern_dict['motion'] = temp
   
  elif(pattern_stratigies_list[i] == 'reduced'):
   temp = np.load('sampling_with_selector_reduced.npy',allow_pickle=True).item()   
   pattern_dict['reduced'] = temp
# 
 correct_instances_from_each_pattern = {}
 pattern_dict_key = list(pattern_dict.keys())
 for key in pattern_dict_key:
  correct_instances_from_each_pattern[key] = {}   
  classes_list = classes_in_each_pattern_dict[key]   
  for class_ in classes_list: 
   correct_instances_from_each_pattern[key][class_] = pattern_dict[key]['correct'][class_]

 false_instances_from_each_pattern = {}
 pattern_dict_key = list(pattern_dict.keys())
 for key in pattern_dict_key:
  false_instances_from_each_pattern[key] = {}   
  classes_list = classes_in_each_pattern_dict[key]   
  for class_ in classes_list:
   false_instances_from_each_pattern[key][class_] = pattern_dict[key]['false'][class_]
 
 
 final_set = {}   
 for key in pattern_dict_key:
   for class_ in classes_in_each_pattern_dict[key]:
    if(class_ not in list(final_set.keys())):   
     final_set[class_] = []  
    class_size = size_of_each_class_in_each_pattern_dict[key][class_]
    print(key)
    print(class_)
    class_accuracy = accuracy_of_each_class_in_each_pattern_dict[key][class_]
    
    for i in range(0,int(class_size*class_accuracy)):
     final_set[class_].extend(correct_instances_from_each_pattern[key][class_][i])  
   
    for i in range(0,int(class_size*(1-class_accuracy))):
     final_set[class_].extend(false_instances_from_each_pattern[key][class_][i])
 return final_set

final_set = single_set_generator(pattern_stratigies_list,classes_in_each_pattern_dict,size_of_each_class_in_each_pattern_dict,accuracy_of_each_class_in_each_pattern_dict) 

np.save('test_assembly.npy', final_set)  