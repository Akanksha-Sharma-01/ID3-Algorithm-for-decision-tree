# ID3-Algorithm-for-decision-tree
This is a course project for course Programming Practicum(CS571) instructed by  Dr. Padmanabhan Rajan and Dr. Siddharth Sarma at IIT Mandi.

Introduction

In this project, a decision tree is designed using ID3 algorithm. A decision tree has nodes and leaf nodes for classification and prediction. ID3 algorithm splits the data into two parts. The splitting of data is done on the basis of information gain which in turn depends on entropy.

In order to design a decision tree using ID3 algorithm, information gain and entropy of each column is calculated. This is done by functions infmation_gain and ent. 

To make a decision tree using ID3 algorithm, one has to find the dominant column or attribute among all columns or attributes of the dataset. This dominant column or attribute now becomes root node for succesive tree. The dominant column is selected on the basis of information gain. The column having high information gain is selected as dominant column. After selecting the dominant column, it is removed from the dataset and a new dataset is created. Same procedure is implemented till we get leaf nodes only. The value of each root node generated is stored in the form of a key of a dictionary. 

In order to apply the decision tree over test data, classify_example function is used. It first split the key of the dictionary and check the condition for the test data. If the conditon leads to leaf node, it classifies the data to a perticular class but if not, it enters to another dictionary and repeats the same process till it reaches to leaf node or a perticular label.

Libraries used

•	Numpy

•	Matplotlib

•	Sklearn (to shuffle the data for training)

•	Pprint (in order to print the dictionary in the form of tree)

Functions used

•	trin_tst_sp(splitData,test_size):

This function splits the data into test and train data.

	splitData = Dataset

	test_size = It is the size of the test data which can be given as number or in percentage


•	check_purity(data):

This function checks whether the given data of same label of class or not.

	data = Any data


•	Clsfy_da(data):

This function classify the label which has maximum number in the data.

	data = Any data  


•	splitfun(data) :

This function returns a dictionary with column number as key and values between max and min of column values as item.

	data =  Data set


•	split_data(data,split_col,split_val):

It is splitting data into two sets. One set contains data of split_col with less than split_val. The other set contains the remaining data of split_col. 

	data = Data set

	split_val = It is the best value for  which information gain of split_col  is max.

	split_col = Dominant column which is going to split into two parts.


•	ent(col):
It calculates the entropy of column which is further used in information gain.
	col = column


•	infmation_gain(data,data_a,data_b):

This function calculates information gain of each column of data set.

	data = Dataset

	data_a, data_b = Splitted data of a particular column 


•	best_col_value(data,col_vis_split):

This function returns best column or best value of the column for which information gain is high.

	col_vis_split = It contains the dictionary of split values of columns.   


•	dec_tree_algor(data,dic2,countt=0,depth=None):

It is the function which makes a decision tree in the form of dictionary.

	data = Dataset

	dic2 = It contains the column names or attributes of the data. 

	depth = It is the number of levels of tree.

	countt = It counts the number of times this recursive function is called.


•	classify_example(example, tree):

This functions classifies a given test data as one of the class.

	example = It is one test data.

	tree = It is the decision tree framed with the training data.


How to use code

The code is easy to apply to any data set. One only needs to change the path of the file or dataset on which the classification is to be done.

The code is able to generate both training and test datasets by its own. One only has to provide either the number of datasets or the percentage of the dataset required in the test data.  

There are two dictionaries which stores the label of the classes and the attributes. These dictionaries can be changed according to the dataset.

By all the mentioned changes, your code is able to form a decision tree and classify on the basis of it.
