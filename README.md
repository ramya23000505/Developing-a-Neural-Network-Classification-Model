# Developing a Neural Network Classification Model
### Date: 27-04-2026
## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="766" height="839" alt="image" src="https://github.com/user-attachments/assets/0d9ab134-2629-4159-a086-25a4f34d24a6" />




## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6:
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: RAMYA R

### Register Number: 212223230169

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self, x):
      #Include your code here
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        
# Initialize model
model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1364" height="264" alt="image" src="https://github.com/user-attachments/assets/2aaeb473-77ed-46d1-a194-dcc788f8d0c5" />


### OUTPUT

## Confusion Matrix

<img width="703" height="596" alt="image" src="https://github.com/user-attachments/assets/5b9afe20-46b8-4a92-975c-a5e34c497924" />


## Classification Report
<img width="748" height="450" alt="image" src="https://github.com/user-attachments/assets/00e6bd83-b2d4-4a69-a67d-a79c4a76d6f3" />


### New Sample Data Prediction
<img width="474" height="105" alt="image" src="https://github.com/user-attachments/assets/ead51a45-8356-4fed-a883-4b9ab58b8c51" />


## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
