#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
from plotly.graph_objs import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data = pd.read_csv('FAO.csv', encoding='latin-1')


# In[9]:


pd.set_option("max_column", 100)


# In[10]:


data.shape


# In[13]:


data.tail(10)


# In[7]:


len(data['Area'].unique())


# In[8]:


data.Element.unique()


# In[9]:


len(data['Element Code'].unique())


# In[10]:


len(data.Item.unique())


# In[11]:


len(data['Item Code'].unique())


# In[12]:


data['Sum Years'] = 0
for year in range(1962, 2014):
    col = 'Y' + str(year)
    data['Sum Years'] = data['Sum Years'] + data[col]


# In[13]:


el_size = data.groupby('Element').agg('size')


# In[14]:


el_size.values


# In[15]:


sns.barplot(el_size.index, el_size.values)
plt.show()


# In[16]:


item_area = []


# In[17]:


for item, group in data.groupby(['Item', 'Area']):
    item_area.append((item[0], item[1], group.Element.values.tolist()))


# In[18]:


only_food = set()
only_feed = set()
food_and_feed = set()


# In[19]:


list(map(lambda x: only_feed.add(x[0]), list(filter(lambda x: 'Food' not in x[2], item_area))));


# In[20]:


list(map(lambda x: only_food.add(x[0]), list(filter(lambda x: 'Feed' not in x[2], item_area))));


# In[21]:


list(map(lambda x: food_and_feed.add(x[0]), list(filter(lambda x: 'Feed' in x[2] and 'Food' in x[2], item_area))));


# In[22]:


only_food.intersection(food_and_feed)


# In[23]:


only_feed.intersection(food_and_feed)


# In[24]:


only_feed.difference(food_and_feed)


# In[25]:


only_food.difference(food_and_feed)


# ### It seems that Cloves, Sesame seed and  Sesameseed Oil are used for food or feed in any country, not both!

# ### The items most produced

# In[26]:


data_item_grouped = data.groupby('Item')


# In[27]:


max_sum_items = data_item_grouped.agg('max')['Sum Years']


# In[28]:


max_sum_items_area = {}


# In[29]:


for item, group in data_item_grouped:
#     print(group[group['Sum Years'] == max_sum_items[item]]['Area'].values[0])
#     print(max_sum_items[item])
    max_sum_items_area[item] = group[group['Sum Years'] == max_sum_items[item]]['Area'].values[0]


# In[30]:


max_sum_items = max_sum_items.to_dict()


# In[31]:


max_sum_items_sorted = sorted(max_sum_items.items(), key=lambda x: x[1], reverse=True)


# In[32]:


titles_areas = []
for k, v in max_sum_items_sorted:
    titles_areas.append(max_sum_items_area[k])


# In[33]:


items = list(map(lambda x: x[0], max_sum_items_sorted))
values = list(map(lambda x: x[1], max_sum_items_sorted))


# In[34]:


titles_areas_items = list(map(lambda x: "(" + x[0] + ")  ,  " + x[1], list(zip(titles_areas, items))))


# In[35]:


fig, ax1 = plt.subplots()
sns.barplot(values[:20], items[:20], ax=ax1)
ax1.tick_params(labeltop=False, labelright=True)
ax_2 = ax1.twinx()
ax_2.set_yticks(list(range(20)))
ax_2.set_yticklabels(titles_areas[:20][::-1])
plt.show()


# ### Change from 1961 to 2013

# In[36]:


area_2013 = data.groupby('Area')['Y2013'].agg('sum')
area_2013


# In[37]:


area_1961 = data.groupby('Area')['Y1961'].agg('sum')
area_1961


# In[38]:


iplot([go.Choropleth(
    locationmode='country names',
    locations=area_1961.index,
    text=area_1961.index,
    z=area_1961.values
)],filename='1961')
iplot([go.Choropleth(
    locationmode='country names',
    locations=area_2013.index,
    text=area_2013.index,
    z=area_2013.values
)],filename='2013')


# In[39]:


data


# In[40]:


data.columns


# In[41]:


def label_encoding(categories):
    categories = list(set(list(categories.values)))
    mapping = {}
    for idx in range(len(categories)):
        mapping[categories[idx]] = idx
    return mapping


# In[42]:


data['Area Abbreviation'] = data['Area Abbreviation'].map(label_encoding(data['Area Abbreviation']))
data.head(10)


# In[43]:


data['Item'] = data['Item'].map(label_encoding(data['Item']))


# In[44]:


data['Area'] = data['Area'].map(label_encoding(data['Area']))
data.head(10)


# In[44]:





# In[45]:


data['Element'] = data['Element'].map(label_encoding(data['Element']))
data.head(10)


# In[46]:


X = data[['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Element Code']].values
y = data[['Element']].values


# In[47]:


from sklearn.model_selection import train_test_split, GridSearchCV
# Splitting Train-set and Test-set
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=41)

# Splitting Train-set and Validation-set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=41)


# In[48]:


from sklearn.naive_bayes import GaussianNB
def get_accuracy(y_true, y_preds):
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_preds).ravel()
    accuracy = (true_positive + true_negative)/(true_negative + false_positive + false_negative + true_positive)
    return accuracy


# In[49]:


naive_b = GaussianNB()
naive_b.fit(X_train, y_train)


# In[50]:


from sklearn.metrics import confusion_matrix

models = [naive_b]
acc = []
for model in models:
    preds_val = model.predict(X_val)
    accuracy = get_accuracy(y_val, preds_val)
    acc.append(accuracy)
    
    
model_name = ['Naive Bayes Accuracy']
accuracy = dict(zip(model_name, acc))
print(accuracy)


# In[51]:


predicted = naive_b.predict(X_val)
cn_matrix = confusion_matrix(y_val, predicted)
print(cn_matrix)


# In[52]:


import seaborn as sns
sns.heatmap(cn_matrix, annot=True)


# In[53]:


from sklearn.metrics import classification_report
predicted = model.predict(X_val)
report = classification_report(y_val, predicted)
print(report)


# In[54]:


from sklearn import datasets, svm
from sklearn.metrics import accuracy_score

x, y = X_train, y_train

clf_predict = svm.SVC(C=7120.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=6.191, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

clf_predict.fit(x, y) 

print("\n",clf_predict.predict(X_test[0:]))
print("\nAccuracy SVM : "+ str(round(accuracy_score(clf_predict.predict(X_test[0:]), y_test[0:])*100, 1)))


# In[55]:


predicted = clf_predict.predict(X_test[0:])
matrix = confusion_matrix(y_test[0:], predicted)
print(matrix)


# In[56]:


import seaborn as sns
sns.heatmap(matrix, annot=True)


# In[57]:


from sklearn.metrics import classification_report
predicted = model.predict(X_test[0:])
report = classification_report(y_test[0:], predicted)
print(report)


# In[57]:




