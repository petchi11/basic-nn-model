# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```python
import pandas as pd
from google.colab import auth
import gspread
from google.auth import default
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('firstdataset').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df.head(n=10)

df.dtypes

df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})

df.dtypes

X = df[['X']].values
Y = df[['Y']].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=50)
X_test.shape

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled

ai_brain = Sequential([
    Dense(2,activation = 'relu'),
    Dense(1,activation = 'relu')
])
ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')
ai_brain.fit(x = X_train_scaled,y = Y_train,epochs = 20000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test
X_test_scaled = scaler.transform(X_test)
X_test_scaled
ai_brain.evaluate(X_test_scaled,Y_test)

input = [[120]]
input_scaled = scaler.transform(input)
input_scaled.shape
input_scaled
ai_brain.predict(input_scaled)
```
## Dataset Information
![image](https://user-images.githubusercontent.com/105230321/194721963-4ccb214f-bcbc-4b25-bb86-3b8f1e41cba5.png)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/105230321/194721971-2f0d2f5c-4c92-407a-80cc-a22fe7d1c331.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/105230321/194721977-e9e029c1-9208-42dc-8a95-f728284a941a.png)

### Sample input and output
![image](https://user-images.githubusercontent.com/105230321/194722009-3540d924-c4dc-465c-92bf-26b414481d17.png)

![image](https://user-images.githubusercontent.com/105230321/194722015-821bb72e-8c21-44e4-b418-e2a113804a4e.png)

## RESULT
Thus, The given dataset is performed with a neural network regression model.
