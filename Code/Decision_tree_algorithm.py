#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


def dec_tree_algor(data,dic2,countt=0,depth=None): #MAIN Algo 
  if( (check_purity(data))or countt==depth ):
    classify=Clsfy_da(data)
    return(classify)
  #recursive part
  else:
    countt+=1
    col_vis_split=splitfun(data)
    split_col,split_val=best_col_value(data,col_vis_split)

    data_B,data_A=split_data(data,split_col,split_val)
    data_B=np.delete(data_B, split_col, 1)
    data_A=np.delete(data_A, split_col, 1)
    
    #sub-tree
    quest="{} <= {}".format(dic2[split_col],split_val)
    sub_tree={quest:[]}
    dic2.remove(dic2[split_col])
    yes_a=dec_tree_algor(data_B,dic2,countt,depth)
    no_a=dec_tree_algor(data_A,dic2,countt,depth)
    
    if yes_a==no_a:
      sub_tree=yes_a
    else:
      sub_tree[quest].append(yes_a)
      sub_tree[quest].append(no_a)
  return(sub_tree)


# In[2]:


# testing 
def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if example[dic3[feature_name]] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


# ### For testing accuracy of testing data set

# In[ ]:


#accuracyof function 
count=0
for i,j in zip(new,test[:,-1]):
  if i==j:
    count=count+1
print(f"the accuracy is {(count/len(test[:,-1]))*100}")

