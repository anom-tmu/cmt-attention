# Import Necessary Packages
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import time
import pickle
from sklearn.preprocessing import StandardScaler

# Fully Connected Neural Networks with One Hidden Layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class Cascade(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Cascade, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(input_size + hidden_size, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2( torch.cat((x,out), dim = 1)  )
        return out


# Device configuration - Open this line when you use CUDA GPU 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters - Choose one model (MLP or Cascade NN) 
# MLP
#input_size = 2
#hidden_size = 3
#output_size = 1

# Cascade
input_size = 2
hidden_size = 1
output_size = 1

# Read file
file_out = pd.read_csv(r"C:\User_Folder\cmt_feature.csv")

x = file_out.iloc[0:1730, 1:3].values    # 5:7     700
y = file_out.iloc[0:1730, 7].values      # 7       700
y = y.reshape(-1, 1)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x) # x
pickle.dump(sc, open('scaler_input.pkl','wb'))

y_train = sc.fit_transform(y) # y
pickle.dump(sc, open('scaler_output.pkl','wb'))

# Converting to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Defining NN model/architechture - Choose one model MLP or Cascade NN
#model = MLP(input_size, hidden_size, output_size)         #.to(device)
model = Cascade(input_size, hidden_size, output_size)      #.to(device)

# Defining Loss and Optimizer
learning_rate = 0.001
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the Model
num_epochs = 10000
steps = len(x_train)
data_loss = []

start_time = time.time()
for epoch in range(num_epochs):

    # Forward pass
    predict = model(x_train)
    loss = criterion(predict, y_train)
        
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # Zero grad before new step
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}],  Loss: {loss.item():.4f}')     # Step [{i+1}/{n_total_steps}],
        data_loss.append(loss.item())
        #print (loss.item())

print("--- %s seconds ---" % (time.time() - start_time))

# Test the Model
i = 0
predict = model(x_train)

np_target = y_train.to('cpu').detach().numpy().copy()
sc_target = sc.inverse_transform(np_target.tolist(), None)

np_predict = predict.to('cpu').detach().numpy().copy()
sc_predict = sc.inverse_transform(np_predict.tolist(), None)

# Count the Error
#np_error = abs(np_predict - np_target)
#ls_error = np_error.tolist()

# Print accuracy of the networks
#max returns (value ,index)    
#acc = 100.0 * n_correct / n_samples
#print(f'Accuracy of the network on test features: {acc} %')

# Save to CSV
np.savetxt("np_loss.csv", data_loss, delimiter=",") 
np.savetxt("np_predict.csv", sc_predict, delimiter=",")        
np.savetxt("np_target.csv", sc_target, delimiter=",") 

# Save the Model
torch.save(model.state_dict(), 'model.pkl')