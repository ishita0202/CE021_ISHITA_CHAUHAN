
# # Part 1 of Lab 1

# import numpy as np

# # 1 Create a Numpy array of size 4 x 5.
#a = np.array([(2,4,6,8,10),(1,3,5,7,9),(1,2,3,4,5),(6,7,8,9,10)])
# print("Shape of numpy array initialized")
# print(a.shape)
# print("\n")

# # 2 Randomly initialize the array. 
# a = np.random.rand(4,5)
# print(a)
# print("\n")

# # 3 Get the Transpose of the Matrix that you created. Create a square matrix and find its determinant.
# print("Transpose of matrix")
# print(a.T)
# print("\n")

# b = np.array([[2,5,3],[4,7,1],[2,6,21]])
# print("Square matrix")
# print(b)
# print("\n")
# print("Determinant of the square matrix")
# x = np.linalg.det(b)
# print(int(x))
# print("\n")

# 4 Create another matrix of size 5 x 4 and randomly initialize it. 
# b = np.array([[1,2,3,4],[8,2,12,23],[10,14,15,3],[12,16,18,13],[11,17,24,27]])
# print("Array of size 5*4")
# print(b)
# print("\n")

# 5 Perform Matrix multiplication. 
# print("Matrix multiplication of 4*5 and 5*4")
# print(np.matmul(a,b))
# print("\n")

# # 6 Perform element wise matrix multiplication.
# b = np.array([(2,4,6,8,10),(1,3,5,7,9),(1,2,3,4,5),(6,7,8,9,10)])
# print("Element wise matrix multiplication")
# print(a*b)
# print("\n")

# # 7 Find mean, median of the numpy array created.
# print("Mean of numpy matrix")
# print(np.mean(a))

# print("Median of numpy matrix")
# print(np.median(a))
# print("\n")

# # 8 Obtain each row in the second column of the first array.


# 9 Convert Numeric entries(columns) of Iris.csv to Mean Centered Version
# columnCentered = a-np.mean(a,axis=0)
# print(columnCentered)
# print("\n")

# 10 Study about numpy array attributes and implement it on the first matrix.
# a = np.arange(1,21).reshape(4,5)
# print("Matrix a using arange")
# print(a)
# print("Dimension of matrix")
# print(a.ndim)
# print("Shape of matrix")
# print(a.shape)
# print("Size of each item")
# print(a.itemsize)
# print("\n")





# # Part 2 Lab 1
import nltk
import re
import string
from nltk.corpus import names
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer


# nltk.download('names')
# nltk.download('stopwords')
# nltk.download('all')

# print("Names in the library :- ")
# print(names.words())
# print("\n")
# print("Number of names :- ")
# print(len(names.words()))
# print("\n")


# names = ['ishita','ishita22222','ishita1234']
# size = [20,50,30]

# plt.pie(size,labels=names,autopct='%.2f%%',shadow=False,startangle=90)
# plt.axis('equal')
# plt.show()

# print("Stop words are :- ")

stopwords_words = stopwords.words('english')
# print(stopwords.words('english'))
# print("Punctuation :- ")
# print(string.punctuation)
# print("\n")

# str = "My name is Ishita Chauhan. 123456789"
# str = re.sub(r'[0-9]','',str)
# print("String without number :- " + str)
# print("\n")

# str = "The @introductory @paragraph, or opening paragraph, is the first paragraph of your essay. It @introduces the main @idea of your essay, captures the @interest of your readers, and tells why your topic is important."
# str = re.sub(r'@','',str)
# print("String without @ :- " + str)
# print("\n")


# str = "The introductory paragraph, or opening paragraph, is the first paragraph of your essay. It introduces the main idea of your essay, captures the interest of your readers, and tells why your topic is important."
# tokenizer = TweetTokenizer(preserve_case=True)
# tokens = tokenizer.tokenize(str)
# # print("Tokenized String :- ")
# # print(tokens)

# stemmer = PorterStemmer()
# ans = []
# for word in tokens:
#     if(word not in stopwords_words and word not in string.punctuation) :
#         ans.append(word)

# print("Tokenized string after removing stopwords and puntuations" )
# print(ans)
# print("\n")



