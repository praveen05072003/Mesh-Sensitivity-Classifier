import torch
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

!pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
!pip install trimesh matplotlib


import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import trimesh
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class NOFFDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith('.off') or f.endswith('.noff'):
                    self.files.append(os.path.join(dirpath, f))
        self.cache = {}

    def len(self):
        return len(self.files)

    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        mesh = trimesh.load(self.files[idx], process=False)

        x = torch.tensor(mesh.vertices, dtype=torch.float)

        edges = []
        for face in mesh.faces:
            edges.append([face[0], face[1]])
            edges.append([face[1], face[2]])
            edges.append([face[2], face[0]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        y = torch.tensor([0], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        self.cache[idx] = data
        return data


dataset_path = "/content/drive/MyDrive/car"
dataset = NOFFDataset(root=dataset_path)

print("Dataset size:", len(dataset))
print(dataset[0])



class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x).squeeze(-1)
        return x



loader = DataLoader(dataset, batch_size=1, shuffle=True)
model = GCN(input_dim=3, hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 6):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")



model.eval()
data = dataset[0].to(device)
data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

with torch.no_grad():
    x1 = F.relu(model.conv1(data.x, data.edge_index))
    x2 = F.relu(model.conv2(x1, data.edge_index))

importance = x2.norm(dim=1).cpu().numpy()

mesh = trimesh.load(dataset.files[0], process=False)
colors = plt.cm.jet((importance - importance.min()) / (importance.max() - importance.min()))[:, :3]
mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
mesh.show()
