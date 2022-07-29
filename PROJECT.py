# In[1]:

import string
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd


# In[4]:


data = pd.read_csv('Car_Purchasing_Data.csv')


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


print("Number of rows",data.shape[0])
print("Nummber of columns",data.shape[1])


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


data.describe()


# In[12]:


import seaborn as sns


# In[13]:


sns.pairplot(data)


# In[14]:


data.columns


# In[15]:


X = data.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)


# In[16]:


y = data['Car Purchase Amount']


# In[17]:


y


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)


# In[20]:


X_scaled


# In[21]:


sc1 = MinMaxScaler()


# In[22]:


y_reshape = y.values.reshape(-1,1)


# In[23]:


y_scaled = sc1.fit_transform(y_reshape)


# In[24]:


y_scaled


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size=0.20,random_state=42)


# In[27]:


data.head()


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# In[29]:


lr = LinearRegression()
lr.fit(X_train,y_train)

svm = SVR()
svm.fit(X_train,y_train)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

xg = XGBRegressor()
xg.fit(X_train,y_train)


# In[30]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense


# In[31]:


ann = Sequential()


# In[32]:


ann.add(Dense(25,input_dim=5,activation='relu'))


# In[33]:


ann.add(Dense(25,activation='relu'))


# In[34]:


ann.add(Dense(1,activation='linear'))


# In[35]:


ann.summary()


# In[36]:


ann.compile(optimizer='adam',loss='mean_squared_error')


# In[37]:


ann.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)


# In[38]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gbr.predict(X_test)
y_pred5 = xg.predict(X_test)
y_pred6 = ann.predict(X_test)


# In[39]:


from sklearn import metrics


# In[40]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
score5 = metrics.r2_score(y_test,y_pred5)
score6 = metrics.r2_score(y_test,y_pred6)


# In[41]:


print(score1,score2,score3,score4,score5,score6)


# In[42]:


final_data = pd.DataFrame({'Models':['LR','SVR','RF','GBR','XG','ANN'],
              'R2_SCORE':[score1,score2,score3,score4,score5,score6]})


# In[43]:


final_data


# In[44]:


import seaborn as sns


# In[45]:


sns.barplot(final_data['Models'],final_data['R2_SCORE'])


# In[46]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense


# In[47]:


ann = Sequential()


# In[48]:


ann.add(Dense(25,input_dim=5,activation='relu'))
ann.add(Dense(25,activation='relu'))
ann.add(Dense(1,activation='linear'))


# In[49]:


ann.compile(optimizer='adam',loss='mean_squared_error')
ann.fit(X_scaled,y_scaled,epochs=100,batch_size=50,verbose=1)


# In[50]:


#  import joblib
import pickle


# In[51]:


# joblib.dump(ann,r'E:\project\car_model')
# pickle.dump(ann, open(r'.\car_model','ab'))  


# In[52]:


# model = joblib.load(r"E:\project\car_model")
model = ann


# In[53]:


import numpy as np


# In[54]:


data.head(1)


# In[55]:


X_test1 = sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))


# In[56]:


X_test1


# In[57]:


pred = ann.predict(X_test1)


# In[58]:


sc1.inverse_transform(pred)


# In[59]:


import numpy as np
from tkinter import*
from sklearn.preprocessing import StandardScaler
import joblib



# In[ ]:


def show_entry_fields():
        
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    
    # model = joblib.load('car_model')
    result = model.predict(sc.transform(np.array([[p1,p2,p3,p4,p5]])))
    Label(master,text='car purchase amount').grid(row=8)
    Label(master,text=sc1.inverse_transform(result)).grid(row=10)
    print("car purchase amount" , sc1.inverse_transform(result)[0][0])
    
    
master = Tk()
master.title("car purchase amount prediction using machine learning ")


label = Label(master,text="car purchase amount prediction using ML"
             ,bg = "black",fg="white").\
             grid(row=0,columnspan=2)



Label(master,text="Gender").grid(row=1)
Label(master,text="Age").grid(row=2)
Label(master,text="Annual Salary").grid(row=3)
Label(master,text="Credit card debt").grid(row=4)
Label(master,text="Net Worth").grid(row=5)

e1=Entry(master)
e2=Entry(master)
e3=Entry(master)
e4=Entry(master)
e5=Entry(master)


e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)


Button(master,text='predict',command=show_entry_fields).grid()

mainloop()


# In[ ]:


from ann_visualizer.visualize import ann_viz;


# In[ ]:


ann_viz(ann,title="ANN")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:













# In[ ]:




