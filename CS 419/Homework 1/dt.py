# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:49:54 2018

@author: BRAHMA
"""

import numpy as np
import csv
import sys
import copy

Node_list = []
id = 0
THRESHOLD1 = 10

class Node:

    def __init__(self, id, name):
        self.id = id
        self.children = []
        self.value = []
        self.threshold = {}
        self.result = -1
        self.name = name

    def add_children(self,children,value):
        self.children.append(children)
        self.value.append(value)
    
    def change_id(self,id):
        self.id = id
        
    def change_result(self,result):
        self.result = result
        
    def change_name(self,name):
        self.name = name
                      
        
def Entropy(S,target_index):
    price = S[:,target_index]
    unique,counts = np.unique(price,return_counts = True)
    probability = counts/np.sum(counts)
    #np.delete(probability,np.where(probability == 0)[0])
    E = -np.sum(np.multiply(probability,np.log2(probability)))
          
    return E
    
def InformationGain(S,A,target_index):
    Gain = Entropy(S,target_index)
    number_S = np.shape(S)
    num_rows_S = number_S[0]
    Attribute = S[:,A]
    unique = np.unique(Attribute)

    for att in unique:
        S_att = S[S[:,A] == att]
        number_S_att = np.shape(S_att)
        num_rows_S_att = number_S_att[0]
        Gain = Gain - (num_rows_S_att/num_rows_S)*Entropy(S_att,target_index)
    return Gain
    
def get_training_data(train):
     with open(train) as csvfile:
        Data = csv.reader(csvfile, delimiter=',')
        temp = list()       
        i = 0
        for row in Data:
            if(i!=0):
                temp.append(row)
            else:
                temp.append(row)
                name = row
            i =i+1
        name_dict = dict((i,n) for n,i in enumerate(name))
        temp = np.array(temp)
        target = temp[:,name_dict['price']]
        data = np.delete(temp,name_dict['id'],1)
        name_dict = dict((i,n) for n,i in enumerate(data[0,:]))
        data = np.delete(data,0,0)
        target = np.delete(target,0,0)
        
        data = data.astype(np.float)
        target = target.astype(np.float)
        
        data = data[:,:]
        target = target[:]
        
        return data, name_dict, target
     
def MakeDecisionTree(data, digital_data, threshold, target, attribute_list, name_dict, depth):
    global id
    unique, count = np.unique(target, return_counts = True)
    n = Node(id,'default')
    n.threshold = threshold
    Node_list.append(n)
    
    if(len(unique) == 1):
        n.result = unique[0] 
        n.name = 'Homogenous'
    elif(not attribute_list):
        n.result = unique[np.argmax(count)]
        n.name = 'InHomogenous'
    elif(depth == 1):
        n.result = unique[np.argmax(count)]
        n.name = 'InHomogenous'
    elif(sum(count) < THRESHOLD1):
        n.result = unique[np.argmax(count)]
        n.name = 'InHomogenous'
    else:
        # Finding the best attribute from tthe attribute list
        Gain = []
        for att in attribute_list:
            Gain.append(InformationGain(digital_data, name_dict[att], name_dict['price']))
        
        max_gain_index = Gain.index(max(Gain))
        max_gain_attribute = attribute_list[max_gain_index]    

        n.change_name(max_gain_attribute)
        unique = np.unique(digital_data[:,name_dict[max_gain_attribute]])
        
        depth = depth - 1
        for value in unique:
            id = id + 1
            n.add_children(id,value)
            indice = (digital_data[:,name_dict[max_gain_attribute]] == value)
            new_data = copy.deepcopy(data[indice])
            new_digital_data, new_threshold = DigitaliseData(new_data,name_dict)
            new_target = copy.deepcopy(target[indice])
            new_attribute_list = copy.deepcopy(attribute_list)
            new_attribute_list.remove(max_gain_attribute)
            MakeDecisionTree(new_data, new_digital_data, new_threshold, new_target, new_attribute_list, name_dict, depth)
    
        id = id + 1
        default_node = Node(id,'default')
        unique, count = np.unique(target, return_counts = True)
        default_node.result = unique[np.argmax(count)]
        Node_list.append(default_node)
        n.add_children(id,None)
        
     
def DigitaliseData(data, name_dict):
    
    digital_data = copy.deepcopy(data)
    threshold_array = {}
    for attribute in name_dict.keys():
        if(attribute != 'price' and attribute != 'zipcode'):
            threshold = np.unique(data[:,name_dict[attribute]])
            IG = []
            for t in threshold:
                indice = data[:,name_dict[attribute]] >= t
                new_data = copy.deepcopy(data)
                new_data[indice,name_dict[attribute]] = 0 
                new_data[np.invert(indice),name_dict[attribute]] = 1
                IG.append(InformationGain(new_data,name_dict[attribute],name_dict['price']))
        
            index = IG.index(max(IG))
            th = threshold[index]
            threshold_array[attribute] = th
            indice = data[:,name_dict[attribute]] >= th
            digital_data[indice,name_dict[attribute]] = 0 
            digital_data[np.invert(indice),name_dict[attribute]] = 1
    
    return digital_data, threshold_array

def Digitalise_row (row, threshold_array, name, name_dict):
    
    digital_row = copy.deepcopy(row)
    if(name != 'zipcode'):
        th = threshold_array[name]
        if(digital_row[name_dict[name]] >= th):
            digital_row[name_dict[name]] = 0
        else:
            digital_row[name_dict[name]] = 1
                    
    return digital_row
        

def evaluate(data, node_list, name_dict):
    
    predicted_price = []

    for row in data:
        result = -1
        node = node_list[0]
        while(result == -1):
            result = node.result
            name = node.name
            #print(name)
            if(name != 'default' and name!='Homogenous' and name!='InHomogenous'):
                digital_row = Digitalise_row(row,node.threshold,name,name_dict)
                value = digital_row[name_dict[name]]
            
            if(len(node.children) != 0):
                if(value in node.value):
                    index = node.value.index(value)
                    child_id = node.children[index]
                    node = node_list[child_id]
                else:
                    index = node.value.index(None)
                    default_id = node.children[index]
                    node = node_list[default_id]
                    
        predicted_price.append(result)
       
    return predicted_price
    
def DecisionTree(train,test,depth):
    
    data, name_dict, target = get_training_data(train)
    digital_data, threshold = DigitaliseData(data, name_dict)
    attribute_list = list(name_dict.keys())
    attribute_list.remove('price')
    MakeDecisionTree(data,digital_data,threshold,target,attribute_list,name_dict, depth)
    
    with open(test) as csvfile:
        Data = csv.reader(csvfile, delimiter=',')
        temp = list()       
        i = 0
        for row in Data:
            if(i!=0):
                temp.append(row)
            else:
                temp.append(row)
                name = row
            i =i+1
        name_dict = dict((i,n) for n,i in enumerate(name))
        temp = np.array(temp)
        data = np.delete(temp,name_dict['id'],1)
        name_dict = dict((i,n) for n,i in enumerate(data[0,:]))
        data = np.delete(data,0,0)
        data = data.astype(np.float)
        data = data[:,:]    
            
    predicted_price = evaluate(data, Node_list, name_dict)
    with open('submission.csv','w',newline='') as csvfile:
        writerr = csv.writer(csvfile,delimiter=',')
        writerr.writerow(['id','price'])
        for i,p in enumerate(predicted_price):
            xid = int(temp[i+1,0])
            p = int(p)
            writerr.writerow([xid,p])
    
    
    print(predicted_price)
    
    #print(np.sum(predicted_price == target)/np.shape(target)[0])
    
    
    
  
S = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,1],[1,1],[1,1],[0,0],[0,0],[0,1],[0,1],[0,1]])
E = Entropy(S,0)
G = InformationGain(S,1,0)
print(G)

          
train = sys.argv[1]
test = sys.argv[2]
depth = int(sys.argv[3])

DecisionTree(train,test,depth)




