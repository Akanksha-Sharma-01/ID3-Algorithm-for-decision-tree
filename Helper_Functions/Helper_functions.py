#!/usr/bin/env python
# coding: utf-8

# In[1]:


def check_purity(data):
  lab_col=data[:,-1]
  uniq=np.unique(lab_col)# it check the purity of data which means that column has only one label or not
  if len(uniq)==1:
    return(True)
  else:
    return(False)
    


# In[2]:


def Clsfy_da(data):
  lab_col=data[:,-1]
  uniq_cls,count_u_cls=np.unique(lab_col,return_counts=True)
  index=count_u_cls.argmax() # classify data 
  clsfy=dic1[uniq_cls[index]]
  return(clsfy)


# In[3]:


def splitfun(data):
  split={}
  n_row,n_col=np.array(data).shape
  for col_ind in range(n_col-1):
    split[col_ind]=[]  #it splits a number based on the maximun value and minimun value of a particular column 
    val=data[:,col_ind]
    mx=max(val)
    mn=min(val)
    lp=np.linspace(mn,mx,50) 
    for ind in lp:
      split[col_ind].append(ind)    
  return(split)


# In[4]:


def split_data(data,split_col,split_val):
  split_colm_val=data[:,split_col]
  data_B=data[split_colm_val<=split_val] #spliting the values on the bases of beat value.

  data_A=data[split_colm_val>split_val]
  return(data_B,data_A)


# In[5]:


#first we are going to calculate entropy of columns so we define the function of entropy 
def ent(col):
  lab_col=col[:,-1]
  _,count=np.unique(lab_col,return_counts=True)
  pro=count/count.sum()
  return(sum(-(pro*np.log2(pro))))


# In[6]:


#To calculate entropy of columns. so we define the function of entropy 
def ent(col):
  lab_col=col[:,-1]
  _,count=np.unique(lab_col,return_counts=True)
  pro=count/count.sum()
  return(sum(-(pro*np.log2(pro))))


# In[7]:


def infmation_gain(data,data_a,data_b):
  comp_ent=ent(data)
  data_1=len(data_a)
  data_2=len(data_b)                 #this is function of information gain
  data_t=sum([data_1,data_2])

  sub_div_ent=((data_1/data_t)*ent(data_a)+(data_2/data_t)*ent(data_b))
  gain=comp_ent-sub_div_ent
  return(gain)


# In[8]:


def best_col_value(data,col_vis_split):
  en_entropy=0
  for col_ind in col_vis_split:
    for val in col_vis_split[col_ind]:
      data_B,data_A=split_data(data,split_col=col_ind,split_val=val)#it select the best col and value on the bases of information gain
      inform_gain=infmation_gain(data,data_A,data_B)
      if inform_gain>en_entropy:
        en_entropy=inform_gain
        best_col=col_ind
        best_val=val
  return(best_col,best_val)



# In[ ]:




