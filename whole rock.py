#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv("/Users/prithwishghosh/Downloads/whole_rock.csv", encoding='latin-1')


# In[4]:


df.head()


# In[5]:


df1 = pd.read_csv("/Users/prithwishghosh/Downloads/complete.csv", encoding='latin-1')


# In[6]:


df1.head()


# In[7]:


df1['rock_origin']


# In[13]:


selected_column = df1['rock_origin']


# In[14]:


df['rock'] = selected_column


# In[15]:


first_chem_rows = df.iloc[:,58:81]


# In[16]:


first_chem_rows['rock'] = df['rock']


# In[17]:


first_chem_rows.head()


# In[18]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[19]:


first_chem_rows.head()


# In[20]:


first_chem_rows = first_chem_rows.fillna("unknown")


# In[21]:


first_chem_rows.head()


# In[22]:


# Assume 'rock_type' is the target column
X = first_chem_rows.drop('rock', axis=1)  # Features (10 chemical components)
y = first_chem_rows['rock']  # Target variable


# In[23]:


first_chem_rows['rock']


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[28]:


# Define the neural network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Predictions on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:





# In[29]:


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[22]:


y_pred_xgboost = model.predict(X_test)


# In[23]:


accuracy = accuracy_score(y_test, y_pred_xgboost)
print(f'Accuracy: {accuracy}')


# In[24]:


new_data = pd.DataFrame({
    'al2o3': [1.83],
    'cr2o3':[1.33],
    'fe2o3':[0.33],
    'fe2o3_tot':[0.53],
    'feo':[10],
    'feo_tot':[4.33],
    'mgo':[10.13],
    'cao':[10.33],
    'mno':[11.33],
    'nio':[12.33],
    'k2o':[13.33],
    'na2o':[11.33],
    'sro':[12.33],
    'p2o5':[21.33],
    'h2o_plus':[13.33],
    'h2o_minus':[0.33],
    'h2o_tot':[8.33],
    'co2':[10.33],
    'so3':[0.33],
    'bao':[0.33],
    'caco3':[0.33],
    'mgco3':[0.33],
    'loi':[20.33],
    # ... add other chemical components as needed
})

# Make predictions on new data
new_predictions = model.predict(new_data)


# In[25]:


predicted_rock_types = [rock_types for label in new_predictions]

# Print the predicted rock types
print("Predicted Rock Types for new data:")
print(predicted_rock_types)


# In[26]:


rock_types = pd.unique(first_chem_rows['rock'])

# Print the unique rock types
print("Unique Rock Types:")
print(rock_types)


# In[27]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[28]:


conf_matrix = confusion_matrix(y_test, y_pred_xgboost)
print('Confusion Matrix:')
print(conf_matrix)


# In[164]:


import numpy as np
np.set_printoptions(threshold=np.inf)  # Set NumPy print options to display the entire array
print('Confusion Matrix:')
print(conf_matrix)


# In[78]:


from sklearn.naive_bayes import GaussianNB


# In[165]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[166]:


y_pred_nb = model.predict(X_test)


# In[167]:


accuracy = accuracy_score(y_test, y_pred_nb)
print(f'Accuracy: {accuracy}')


# In[168]:


conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print('Confusion Matrix:')
print(conf_matrix_nb)


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[148]:


plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')


# In[147]:


plt.figure(figsize=(40, 20))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.show()
plt.savefig('confusion_matrix_nb.png')


# In[170]:


latex_code = pd.DataFrame(conf_matrix).to_latex(index=False, header=False)

# Save the LaTeX code to a file
with open('confusion_matrix_xg.tex', 'w') as f:
    f.write(latex_code)


# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)


# In[ ]:


svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)


# In[ ]:


y_pred = svm_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix


# In[ ]:


accuracy


# In[174]:


# Example model names and accuracies
model_names = ['XGBoost', 'Naive Bias', 'LDA', 'Bayesian Discriminant Analysis']
accuracies = [0.6203, 0.4933, 0.5172, 0.5219]

# Create a bar plot
plt.figure(figsize=(18, 6))
plt.bar(model_names, accuracies, color='red')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.ylim(0, 1)  # Set y-axis limits (0 to 1 for accuracy)
plt.show()


# In[35]:


import matplotlib.pyplot as plt

# Machine learning algorithms
algorithms = ['XGBoost', 'Naive Bias', 'LDA', 'Bayesian Discriminant Analysis']

# Accuracy scores
accuracy_scores = [0.6203, 0.4933, 0.5172, 0.5219]

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(algorithms, accuracy_scores, color='skyblue')
plt.xlabel('Accuracy Score')
plt.title('Accuracy of Different Machine Learning Algorithms')
plt.xlim(0, 1)  # Adjust the x-axis limits if needed
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

