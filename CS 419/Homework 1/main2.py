
# coding: utf-8

# In[1]:


import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd


# In[2]:


data = pd.read_csv('./train.csv')
MAX_DEPTH = 250
MIN_NODE_LENGTH = 1
data.head()


# In[3]:


def MSE(x):
    if len(x) == 0:
        return 0
    x_mean = np.mean(x)
    mse = np.mean(np.square(x-x_mean))
    return sum(np.square(x-x_mean))


# In[4]:


def absoluteError(x):
    if len(x) == 0:
        return 0
    x_median = np.median(x)
    sum = 0
    for i in x:
        sum += np.abs(i - x_median)
    return sum


# In[5]:


errorFunctions = {'mse':MSE, 'absolute':absoluteError}


# In[6]:


def normalize(mat):
    a = np.array(mat)
    a = a.astype(np.float64)
    b = np.apply_along_axis(lambda x: (x - np.min(x)) /
                            float(np.max(x) - np.min(x)), 0, a)
    return b


# In[7]:


data = np.array(data).astype('float64')
np.random.shuffle(data)

split = int(0.69 * data.shape[0])

X = data[:split,:-1]
y = data[:split,-1]

data_val = data[split:,:]

leaves = []

X = normalize(X)


# In[8]:


def splitValueSelection(a, Y, errorMetric = 'mse'):
    #select error function
    error_func = errorFunctions[errorMetric]
    parentError = error_func(Y)
    a = np.copy(a)
    Y = np.copy(Y)
    #sort a,Y according to a
    sort_a = a.argsort()
    a = a[sort_a]
    Y = Y[sort_a]
    
    #init max vals
    max_t = a[0]
    maxGain  = -np.inf
    
    # iterate over var for
    i=0
    while i < len(a):
        while i<len(a)-1 and a[i]==a[i+1]:
            i += 1
        
        #leftList = a[:i+1]
        #rightList = a[i+1:]
        leftY = Y[:i]
        rightY = Y[i:]
        
        leftError = error_func(leftY)
        rightError = error_func(rightY)
        
        infGain = parentError - (i* leftError + (len(a)-i) * rightError)/len(a)
        
        if infGain >= maxGain:
            maxGain = infGain
            max_t = a[i]

        i += 1
    if maxGain == -np.inf:
            print('YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')

    return max_t, maxGain
    return None, None


# In[9]:


def Entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    relativeFrequency = counts/sum(counts)
    entropy = -sum(np.multiply(relativeFrequency, np.log2(relativeFrequency)))
    return entropy


# In[10]:


def bestAttributeSplit(X,y):
    maxInfGain = -np.inf
    good_split = False
    index = None
    for i in range(X.shape[1]):
        infGain = splitValueSelection(X[:,i], y)[1]
        if infGain == None:
            continue
        if infGain > maxInfGain:
            good_split = True
            maxInfGain = infGain
            index = i
    if not good_split:
        return -1
    return index


# In[11]:


class Node:
    
    def __init__(self, depth, X, y, data_val):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.depth = depth
        self.y_mean = np.mean(self.y)
        #self.threshold = 1
        self.is_leaf = False
        self.assignAttr()
        self.data_val = np.copy(data_val)
        
        
        
    def assignChildren(self, child_left, child_right):
        child_left.parent = self
        child_right.parent = self
        self.leftChild = child_left
        self.rightChild = child_right

    def assignAttr(self):
        self.attributeIndex = bestAttributeSplit(self.X, self.y)
        if self.attributeIndex == -1:
            self.is_leaf = True
            self.threshold = None
            self.infGain = None
        else:
            self.threshold, self.infGain = splitValueSelection(self.X[:,self.attributeIndex], self.y)    
            
            
            
    def check_with_threshold(self, x):
        #print(self.attributeIndex, x)
        return x[self.attributeIndex] >= self.threshold
           
    def splitValidationData(self):
        self.rightData = np.array([])
        self.leftData = np.array([])
        for i in range(self.data_val.shape[0]):
            if self.check_with_threshold(data_val[i]):
                np.append(self.rightData,data_val[i])
            else:
                np.append(self.leftData,data_val[i])
        if self.leftData.ndim == 1:
            self.leftData = self.leftData.reshape(-1,1)
        if self.rightData.ndim == 1:
            self.rightData = self.rightData.reshape(-1,1)
        
    def checkLeaf(self):
        if self.depth >= MAX_DEPTH or len(self.y) <= MIN_NODE_LENGTH or len(np.unique(self.y)) == 1 or self.is_leaf:
            self.is_leaf = True
            leaves.append(self)
        return self.is_leaf


# In[12]:


def makeDecisionTree(node):
    if node.checkLeaf():
        print(node.attributeIndex, node.threshold,node.y_mean, node.depth)
        return
    
    X = np.copy(node.X)
    y = np.copy(node.y)
    bestAttribute = node.attributeIndex
    
    sort_X = X[:,bestAttribute].argsort()
    X = X[sort_X]
    y = y[sort_X]
    split_index = np.searchsorted(X[:,bestAttribute], node.threshold)
    
    #print('TEST', node.threshold in X[:,bestAttribute], node.threshold)
    
    X1, X2 = X[:split_index,:], X[split_index:,:]
    y1, y2 = y[:split_index], y[split_index:]
    #print('splitIndex', split_index)
    
    
    
    if len(y1) > 0 and len(y1) < len(y):
        
        node.splitValidationData()
        
        node1 = Node(node.depth+1, X1, y1, node.leftData)
        node2 = Node(node.depth+1, X2, y2, node.rightData)
    
        node.assignChildren(node1, node2)

        makeDecisionTree(node1)
        makeDecisionTree(node2)
    else:
        print('here')
        node.is_leaf = True
        makeDecisionTree(node)
    return


# In[15]:


rootNode = Node(0, X, y, data_val)
rootNode.parent = rootNode
makeDecisionTree(rootNode)


# In[16]:


print('Decision Tree created')
print('No. of leaves =', len(leaves))


# In[17]:


def prune_condition(node, error_metric='mse'):
    parent = node
    error_func = errorFunctions[error_metric]
    parentError = error_func(parent.data_val[:,-1])
    leftError = error_func(parent.leftChild.data_val[:,-1])
    rightError = error_func(parent.rightChild.data_val[:,-1])
    if parent.data_val.shape[0] == 0:
        infGain = 0
    else:
        print('yooo')
        infGain = parentError                - (parent.leftChild.data_val.shape[0] * leftError                    + (parent.rightChild.data_val.shape[0]) * rightError)/parent.data_val.shape[0]
    #print(infGain)
    return infGain


# In[18]:


def error(y1, x):
    if len(y1) == 0:
        return 0
    if x.ndim > 1:
        y2 = np.array([predict(i) for i in x])
    else:
        y2 = predict(x)
    sum_error = np.square(y1-y2)
    return np.mean(sum_error)


# In[19]:


def prune(leaves, count):
    if len(leaves) < 2:
        print(count)
        return count
    parents = []
    for leaf in leaves:
        leaf = leaf.parent
        if leaf.depth == 1:
            print(count)
            return count

        parent = leaf.parent
        l = parent.leftChild
        r = parent.rightChild
        if error(parent.data_val[:,:-1],parent.data_val[:,-1]) <= (len(l.y) * error(l.data_val[:,:-1],l.data_val[:,-1]) + len(r.y) * error(r.data_val[:,:-1],r.data_val[:,-1]))/len(parent.y):
            parent.is_leaf = True
            if parent not in parents:
                count += 1
                parents.append(leaf.parent)
    return prune(parents, count)


# In[20]:


p = prune(leaves, 0)
print(p)


# In[21]:


X_test = np.array(pd.read_csv('test.csv'))
X_test = normalize(X_test)


# In[22]:


def predict(x):
    node = rootNode
    while True:
        if node.is_leaf:
            return node.y_mean
        if (node.check_with_threshold(x)):
            node = node.rightChild
        elif (not node.check_with_threshold(x)):
            node = node.leftChild
        else:
            print('ERROR', node.nChild)


# In[23]:


def checkTree(n):
    x = X[n]
    ans = y[n]
    print(predict(x), ans)


# In[24]:


print(error(y,X))


# In[25]:


checkTree(np.random.randint(0,X.shape[0]))


# In[537]:


f = open('output.csv','w')
f.write('Id,quality\n')
for i in range(X_test.shape[0]):
    f.write(str(i+1)+','+str(predict(X_test[i]))+'\n')
f.close()


# In[514]:


#leaves.sort(key = lambda x: x.y_mean)

