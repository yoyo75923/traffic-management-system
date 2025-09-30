import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class TrafficGNN(nn.Module):
    def __init__(self):
        super(TrafficGNN, self).__init__()
        self.conv1 = pyg_nn.GraphConv(16, 32)
        self.conv2 = pyg_nn.GraphConv(32, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a dataset class for the traffic data
class TrafficDataset(pyg_data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data

# Create a data loader for the traffic data
data_list = [...]  # list of traffic data
dataset = TrafficDataset(data_list)
data_loader = pyg_data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train the graph neural network
model = TrafficGNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for data in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
