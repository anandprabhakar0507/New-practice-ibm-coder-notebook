
# coding: utf-8

# ### Import the JSON File with Industry Lab Sensor Data

# In[1]:



import sys
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_915ea66450e44183938b1aab8572887f = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='x8LKJUWfwBvzMvyp4glOaZk6VQOcrYmbWV80lKToccNW',
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_915ea66450e44183938b1aab8572887f.get_object(Bucket='dsbootcampac3431d743f2492ebe1cfe6103674873',Key='floorsensordata2604.json')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object 

if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

# Since JSON data can be semi-structured and contain additional metadata, it is possible that you might face an error during data loading.
# Please read the documentation of 'pandas.read_json()' and 'pandas.io.json.json_normalize' to learn more about the possibilities to adjust the data loading.
# pandas documentation: http://pandas.pydata.org/pandas-docs/stable/io.html#io-json-reader
# and http://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.json.json_normalize.html

df_data_1 = pd.read_json(body, orient='values')
df_data_1.head()



# ### Check the Structure of the Data Frame

# In[4]:


df = df_data_1


# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.itemname.unique()


# ### Format Yanzi Sensor Data 

# In[8]:


df['id'] = df['itemname'].str.split('_').str[0]


# In[9]:


df['temperature'] = df[df['itemname'].str.contains('temperature')]['value'].astype(float)
df['carbonDioxide'] = df[df['itemname'].str.contains('carbonDioxide')]['value'].astype(float)
df['humidity'] = df[df['itemname'].str.contains('humidity')]['value'].astype(float)
df['illuminance'] = df[df['itemname'].str.contains('illuminance')]['value'].astype(float)
df['pressure'] = df[df['itemname'].str.contains('pressure')]['value'].astype(float)
df['Occupancy'] = df[df['itemname'].str.contains('Occupancy')]['value']


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.describe()


# ### Visualize Data with Pixidust

# In[13]:


import pixiedust


# In[16]:


display(df)


# ### Filter on Shinano and Fill Missing Values

# In[17]:


df_shinano = df[df['id'].isin(['4674C', 'Shinano'])].copy()


# In[18]:


df_shinano.sort_values(by='time', inplace=True)


# In[19]:


df_shinano.index = pd.to_datetime(df_shinano.time)


# In[20]:


df_shinano.fillna(method='ffill', inplace=True)


# In[21]:


df_shinano.fillna(method='bfill', inplace=True)


# In[22]:


df_shinano.head()


# In[23]:


df_shinano.drop(['illuminance'], axis=1, inplace=True)


# In[24]:


df_shinano.describe()


# In[25]:


df_shinano.Occupancy.unique()


# In[26]:


df_shinano.groupby('Occupancy')['time'].nunique()


# ### Plot Sensor Values depending on Occupancy State

# In[27]:


import matplotlib.pyplot as plt


# In[28]:


df_shinano.temperature.plot()


# ### Build Simple Prediction Model

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


df_shinano_model = df_shinano.copy()


# In[31]:


df_shinano_model.loc[df_shinano_model.Occupancy == 'free', 'Occupancy'] = 0
df_shinano_model.loc[df_shinano_model.Occupancy == 'occupied', 'Occupancy'] = 1


# In[32]:


y = df_shinano_model['Occupancy'].values
y


# In[33]:


X = df_shinano_model.loc[:, ['temperature', 'carbonDioxide', 'humidity', 'pressure']].values
X


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve


# In[36]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[37]:


clf = RandomForestClassifier(n_estimators=10)


# In[38]:


clf.fit(X_train, y_train)


# In[39]:


clf.score(X_test, y_test)


# In[40]:


predictions = clf.predict(X_test)


# In[41]:


print(classification_report(y_test, predictions))


# In[42]:


title = 'Learning Curves (Random Forest)'
estimator = clf
plot_learning_curve(estimator, title, X_train, y_train)
plt.show()

