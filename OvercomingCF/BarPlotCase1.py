import numpy as np
import matplotlib.pyplot as plt
import pickle


    
with open('NewInstanceCase1_lr0.01_E5_B16_EachSet100', 'rb') as handle:
    op = pickle.load(handle)

with open('NewInstanceCase1_lr0.01_E5_B16_EWC_EachSet100', 'rb') as handle:
    EWCop = pickle.load(handle)


op_data = op[20] 
EWCop_data = EWCop[20]
initial = op[0]

#initial = initial[1:]
#op_data = op_data[1:]
#EWCop_data = EWCop_data[1:]


labels = ['O','F0', 'F1', 'F0', 'F1', 'F0', 'F1', 'F0', 'F1', 'F0', 'F1','F0', 'F1', 'F0', 'F1', 'F0', 'F1', 'F0', 'F1', 'F0', 'F1']
plt.xticks(range(21), labels)
width =0.15
plt.bar(np.arange(len(initial)), initial, width=width,label='inital')
plt.bar(np.arange(len(op_data))+ width, op_data, width=width, label = 'op_data')
plt.bar(np.arange(len(EWCop_data))+ width*2, EWCop_data, width=width, label = 'EWCop_data')
plt.legend()
plt.show()
