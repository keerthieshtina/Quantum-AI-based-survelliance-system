
# ---------------- 2) Settings ----------------
import os, time
DATASET_ROOT = r"G:\My Drive\dataset1\ucf_anamoly_subset"
MAX_SAMPLES = 2000
IMG_SIZE = 32
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 15        
PRINT_EVERY = 10
VAL_SPLIT = 0.2
NUM_QUBITS = 2

# ---------------- 3) Dataset & Augmentation ----------------
import cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.4):  # noise for regularization
        self.mean = mean; self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4,0.4,0.4,0.2),
    transforms.RandomRotation(25),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.4),
])

valid_exts = (".mp4",".avi",".mov",".mkv",".jpg",".png",".jpeg",".tif",".tiff")
media, classes, class_idx = [], {}, 0
print(" Scanning dataset...")
for root,_,files in os.walk(DATASET_ROOT):
    for f in files:
        if f.lower().endswith(valid_exts):
            folder = os.path.basename(root) or "root"
            if folder not in classes:
                classes[folder] = class_idx; class_idx += 1
            media.append((os.path.join(root,f), classes[folder]))
if MAX_SAMPLES and len(media) > MAX_SAMPLES:
    media = media[:MAX_SAMPLES]
print(f"Total media files: {len(media)}, Classes: {classes}")

class FastMediaDataset(Dataset):
    def __init__(self, items, transform=None): self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg",".jpeg",".png",".tif",".tiff"):
            img = cv2.imread(path)
            img = np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8) if img is None else cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            cap = cv2.VideoCapture(path); success, frame = cap.read(); cap.release()
            frame = np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8) if not success else frame
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img,label

dataset = FastMediaDataset(media, transform=transform)
n_total = len(dataset); n_val = int(n_total*VAL_SPLIT); n_train = n_total-n_val
train_ds,val_ds = random_split(dataset,[n_train,n_val])
train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
print("Train samples:", n_train,"Validation samples:", n_val)

# ---------------- 4) Hybrid Quantum Model ----------------
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

input_params = [Parameter(f"x{i}") for i in range(NUM_QUBITS)]
qc = QuantumCircuit(NUM_QUBITS)
for i in range(NUM_QUBITS): qc.ry(input_params[i],i)
for i in range(NUM_QUBITS-1): qc.cx(i,i+1)

sampler_qnn = SamplerQNN(circuit=qc,input_params=input_params,weight_params=[],sampler=Sampler())
q_layer = TorchConnector(sampler_qnn)

class QuantumAttention(nn.Module):
    def __init__(self,input_dim,num_qubits):
        super().__init__()
        self.qnn = q_layer
        self.fc_in, self.fc_out = nn.Linear(input_dim,num_qubits), nn.Linear(2**num_qubits,input_dim)
        self.norm = nn.LayerNorm(input_dim)
    def forward(self,x):
        q_in = self.fc_in(x); q_out = self.qnn(q_in)
        out = self.fc_out(q_out)
        return self.norm(x+out)

class HybridQuantumTransformer(nn.Module):
    def __init__(self,n_qubits,n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,4,3,stride=2,padding=1),
            nn.BatchNorm2d(4), nn.ReLU(),
            nn.Conv2d(4,8,3,stride=2,padding=1),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 8*(IMG_SIZE//4)*(IMG_SIZE//4)
        self.q_attn = QuantumAttention(conv_out,n_qubits)
        self.dropout = nn.Dropout(0.7)   # stronger dropout
        self.classifier = nn.Sequential(
            nn.Linear(conv_out,16), nn.ReLU(), nn.Dropout(0.7),  # smaller hidden layer
            nn.Linear(16,n_classes)
        )
    def forward(self,x):
        x_conv = self.conv(x)
        x_attn = self.q_attn(x_conv)
        x = self.dropout(x_conv+x_attn)
        return self.classifier(x)

n_classes = max(classes.values())+1
model = HybridQuantumTransformer(NUM_QUBITS,n_classes)

# ---------------- 5) Training Setup ----------------
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-2)  # smaller lr + stronger reg
criterion = nn.CrossEntropyLoss()

def evaluate(loader):
    model.eval(); all_preds,all_labels,total_loss=[],[],0.0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            outputs = model(X); loss = criterion(outputs,y)
            total_loss += loss.item()*X.size(0)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss/len(loader.dataset)
    acc = accuracy_score(all_labels,all_preds)
    prec,rec,f1,_ = precision_recall_fscore_support(all_labels,all_preds,average='weighted',zero_division=0)
    return avg_loss, acc, prec, rec, f1

# ---------------- 6) Training Loop ----------------
if __name__ == "__main__":
    print(f" Training on {n_train} samples...")
    for epoch in range(NUM_EPOCHS):
        model.train(); running_loss=0.0
        for batch_idx,(X,y) in enumerate(train_loader,1):
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X),y)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            if batch_idx % PRINT_EVERY == 0:
                print(f"[Epoch {epoch+1} Batch {batch_idx}] Loss: {running_loss/batch_idx:.4f}")

        train_loss,train_acc,train_prec,train_rec,train_f1 = evaluate(train_loader)
        val_loss,val_acc,val_prec,val_rec,val_f1 = evaluate(val_loader)
        print(f"Epoch {epoch+1}:")
        print(f"  Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2%}, Prec: {train_prec:.2%}, Rec: {train_rec:.2%}, F1: {train_f1:.2%}")
        print(f"  Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, Prec: {val_prec:.2%}, Rec: {val_rec:.2%}, F1: {val_f1:.2%}")
        # ---------------- 7) Save Model ----------------
    torch.save(model.state_dict(), "hybrid_quantum_model.pth")

print(" Model saved successfully!")

    