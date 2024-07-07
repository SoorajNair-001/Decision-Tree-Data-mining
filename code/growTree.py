import json
import numpy as np

# function to find the entropy
def entropy(low,high):
    if(low == 0 or high == 0):
        return 0
    count = low+high
    entropy1 = -((low/count) * np.log2((low/count)))
    entropy2 = -((high/count) * np.log2((high/count)))
    return entropy1+entropy2

# recursive function to calculate the tree after root node
def calculate_nextLvl(data_interpret, training_Set, Node_index, rootNode_index, risk, closed, parent_list,encode):
    # encoding
    encode.append("[")
    encode.append('"')
    encode.append(data_interpret[Node_index][0])
    encode.append('"')
    encode.append(",")
    prevClosed = closed.copy() # previous closed list
    count = 0
    # for each value of the attribute
    for val_root in data_interpret[Node_index][1]:
        count +=1
        if(count>1):
            closed = prevClosed.copy()
        if(count == 1):
            encode.append("{")
        encode.append('"')
        encode.append(val_root)
        encode.append('"')
        encode.append(":")
        # finding current list
        S = []
        for i in range(len(risk)):
            found = True
            if(training_Set[Node_index][i] != val_root):
                found = False
            for parent in parent_list:
                if(training_Set[parent[0]][i] != parent[1]):
                    found = False
            if(found == True):
                S.append(risk[i])
        S = np.array(S)
        # counting number of low and high in current list
        low_Risk_count = np.count_nonzero(S == 1)
        high_Risk_count = np.count_nonzero(S == 2)
        # if leaf node, stop and continue with the rest
        if (low_Risk_count == 0 and high_Risk_count == 0):
            low = parent_list[len(parent_list)-1][2]
            high = parent_list[len(parent_list)-1][3]
            if(low>high):
                encode.append(1)
            else:
                encode.append(2)
            if (count != len(data_interpret[Node_index][1])):
                encode.append(",")
            continue
        if(low_Risk_count == 0):
            encode.append(2)
            if(count != len(data_interpret[Node_index][1])):
                encode.append(",")
            continue
        if(high_Risk_count == 0):
            encode.append(1)
            if (count != len(data_interpret[Node_index][1])):
                encode.append(",")
            continue
        # calculating entropy of current
        entropy_S = entropy(low_Risk_count,high_Risk_count)
        index = 0
        gain_of_attrs = []
        for attr in data_interpret:
            if(attr[0] not in closed):
                attr_entropy_list = [] # list to store entropy values
                attr_data = np.array(training_Set[index]) # current node data
                root_node = np.array(training_Set[Node_index]) # root node data
                # for each value in attribute
                for val in attr[1]:
                    count_low = 0 # number of lows
                    count_high = 0 # number of highs
                    # count number lows and highs
                    for i in range(len(attr_data)):
                        if(attr_data[i] == val and root_node[i]==val_root and risk[i] == 1):
                            count_low += 1
                        if(attr_data[i] == val and root_node[i]==val_root and risk[i] == 2):
                            count_high += 1
                    # calculate entropy
                    attr_entropy_list.append([entropy(count_low,count_high),count_low,count_high])
                entropy_Sattr = 0
                # calculate gain
                for x in attr_entropy_list:
                    entropy_Sattr += ((x[1]+x[2])/len(risk))*x[0]
                gain_attr = entropy_S - entropy_Sattr
                gain_of_attrs.append(gain_attr)
                index += 1
            else: 
                gain_of_attrs.append(0)
                index += 1
        # node with maximum gain selected as next parent
        max_gain = np.max(gain_of_attrs)
        nextNode = np.where(gain_of_attrs == max_gain)
        nextNode_index = int(nextNode[0][0])
        if(nextNode_index != 0):
            closed.append(data_interpret[nextNode_index][0]) # new parent added to closed list
            parent_list.append([Node_index,val_root,low_Risk_count,high_Risk_count]) # new parent added to parents list
            # recuresive call
            calculate_nextLvl(data_interpret,training_Set,nextNode_index,rootNode_index,risk,closed,parent_list,encode)
            parent_list.pop()
            encode.append("}")
            encode.append("]")
            encode.append(",")
        else:
            # root node if no parent found
            if(low_Risk_count > high_Risk_count):
                encode.append(1)
            else:
                encode.append(2)
            encode.append(",")


# main funcrion
def main():
    # loading data_interpret.txt
    with open('../data/dataDesc.txt') as f: 
        data_interpret = json.load(f)
    file = open('../data/train.txt', "r")  # reading the Train.txt file
    training_Set = []  # 
    # appending values to Training set Matrix
    for x in file.readlines():
        row = x.split(' ')
        intRow = [int(ele) for ele in row]
        training_Set.append(intRow)
        
    risk = np.array(training_Set[0])
    closed = ['RISK'] # closed list to store checked attributes
    
    # RISK - class label
    low_Risk_count = np.count_nonzero(risk == 1)
    high_Risk_count = np.count_nonzero(risk == 2)
    entropy_S = entropy(low_Risk_count,high_Risk_count)

    gain_of_attrs = []
    index = 0
    # Finding root node
    for attr in data_interpret:
        if(attr[0] not in closed): # going through all attributes
            attr_entropy_list = [] # list to store all entropies
            attr_data = np.array(training_Set[index]) # current attribute data
            # for each value of the attribute
            for val in attr[1]:
                # calculate number of high and low
                count_low = 0
                count_high = 0
                for i in range(len(attr_data)):
                    if(attr_data[i] == val and risk[i] == 1):
                        count_low += 1
                    if(attr_data[i] == val and risk[i] == 2):
                        count_high += 1
                # finding entropy and storing in list
                attr_entropy_list.append([entropy(count_low,count_high),count_low,count_high])
            entropy_Sattr = 0
            # finding gain of the attribute
            for x in attr_entropy_list:
                entropy_Sattr += ((x[1]+x[2])/len(risk))*x[0]
            gain_attr = entropy_S - entropy_Sattr
            gain_of_attrs.append(gain_attr)
            index += 1
        else: 
            gain_of_attrs.append(0)
            index += 1

    # the attribute with maximum gain is selected as root node
    max_gain = np.max(gain_of_attrs)
    rootNode = np.where(gain_of_attrs == max_gain)
    rootNode_index = int(rootNode[0][0])
    closed.append(data_interpret[rootNode_index][0]) # adding root node to closed list

    encode_list = [] # list to store encoded tree
    parent_list = [] # list to store the parents
    # recursive function to find the rest of the tree
    calculate_nextLvl(data_interpret,training_Set,rootNode_index,rootNode_index,risk,closed,parent_list,encode_list)
    encode_list.append("}")
    encode_list.append("]")
    # encoded list
    newEncode_list = []
    for ele in range(0,len(encode_list)):
        if(not(encode_list[ele] == "," and encode_list[ele+1] == "}")):
            newEncode_list.append(encode_list[ele])
    # encode the list as string
    encoded_tree = ''.join(map(str, newEncode_list))
    # store encoded tree to file
    with open('../data/fullTree.txt', 'w') as f:
        f.write(encoded_tree)
    # return name of the file where the tree is saved
    return "fullTree.txt"
