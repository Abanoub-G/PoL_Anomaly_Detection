import numpy as np
import matplotlib.pyplot as plt
import pickle


#with open('NewInstanceCase2_Plain_EachSet100FromX_train', 'wb') as fp:
#    pickle.dump(acc_list_after_each_training, fp)

with open('NewInstanceCase2_Plain_EachSet100FromX_train', 'rb') as handle:
    plain = pickle.load(handle)
    
with open('NewInstanceCase2_Op_EB_Hyper_EachSet100FromX_train', 'rb') as handle:
    op = pickle.load(handle)

with open('NewInstanceCase2_EWCOp_EB_Hyper_EachSet100FromX_train', 'rb') as handle:
    EWCop = pickle.load(handle)


plain_data = plain[10]
op_data = op[10] 
EWCop_data = EWCop[10]
initial = plain[0]

#initial = initial[1:]
#op_data = op_data[1:]
#EWCop_data = EWCop_data[1:]


labels = ['Orignal','N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
plt.xticks(range(11), labels)
width =0.15
plt.bar(np.arange(len(initial)), initial, width=width,label='inital')
plt.bar(np.arange(len(op_data))+ width, op_data, width=width, label = 'op_data')
plt.bar(np.arange(len(EWCop_data))+ width*2, EWCop_data, width=width, label = 'EWCop_data')
plt.legend()
plt.show()


