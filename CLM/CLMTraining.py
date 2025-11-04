# ============================================================
# CLM Training.py — Enhanced LSTM + SI + FIM (adaptive λ, EMA)
# Prints R² (eval/re-eval), Forgetting, Memory Stability; draws plots
# ============================================================

# ============================================================
# Enhanced LSTM + SI + FIM (adaptive λ, EMA, 2-layer LSTM)
# Added R² Initial vs Final, Forgetting, and Memory Stability
# ============================================================

import torch, torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np, pandas as pd, random, time, warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

class SimpleDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __getitem__(self, i): return self.X[i], self.y[i]
    def __len__(self): return len(self.X)

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.fc(out).squeeze()

class SI_Tracker:
    def __init__(self, model, damping=0.1):
        self.model = model; self.damping = damping
        self.prev = {n:p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
        self.omega = {n:torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
        self.w = {n:torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    def accumulate(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.w[n] += p.grad.detach() * (p.detach() - self.prev[n])
    def update(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                delta = p.detach() - self.prev[n]
                self.omega[n] += torch.abs(self.w[n]) / (delta**2 + self.damping)
                self.w[n].zero_(); self.prev[n] = p.detach().clone()
    def penalty(self):
        return sum((self.omega[n] * (p - self.prev[n])**2).sum()
                   for n,p in self.model.named_parameters() if p.requires_grad)

def compute_FIM(model, loader, crit):
    F = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    for X, y in loader:
        out = model(X); loss = crit(out, y); loss.backward()
        for n,p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                F[n] += p.grad.detach()**2
        model.zero_grad()
    for n in F:
        F[n] /= max(1,len(loader))
        F[n] /= (F[n].mean().abs() + 1e-6)
    return F

def prepare_data(path, num_tasks=10, test_size=0.2):
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c not in ["value", "Date"]]
    size = len(df)//num_tasks
    tasks=[]
    for i in range(num_tasks):
        chunk=df.iloc[i*size:(i+1)*size]
        sx, sy = StandardScaler(), StandardScaler()
        X = torch.FloatTensor(sx.fit_transform(chunk[feat].values))
        y = torch.FloatTensor(sy.fit_transform(chunk[["value"]].values).flatten())
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42)
        tasks.append((DataLoader(SimpleDataset(Xtr,ytr),batch_size=32,shuffle=True),
                      DataLoader(SimpleDataset(Xte,yte),batch_size=32,shuffle=False), sy))
    return tasks, len(feat)

def evaluate(model, loader, scaler):
    model.eval(); p,t=[],[]
    with torch.no_grad():
        for X,y in loader:
            o=model(X); p.extend(o.numpy()); t.extend(y.numpy())
    p,t=np.array(p),np.array(t)
    p=scaler.inverse_transform(p.reshape(-1,1)).flatten()
    t=scaler.inverse_transform(t.reshape(-1,1)).flatten()
    return max(0,r2_score(t,p))

def hybrid_train(path, num_tasks=10):
    tasks,input_dim = prepare_data(path,num_tasks)
    print(f"=== Enhanced LSTM+SI+FIM on {path.split('/')[-1]} ===")
    model=LSTM_Model(input_dim)
    crit=nn.MSELoss()
    opt=torch.optim.Adam(model.parameters(),lr=0.0003)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
    si=SI_Tracker(model); FIM_prev=None
    ema_decay=0.95

    # Warmup
    allX,allY=[],[]
    for tr,_,_ in tasks:
        for X,y in tr: allX.append(X); allY.append(y)
    allX=torch.cat(allX); allY=torch.cat(allY)
    warm_loader=DataLoader(SimpleDataset(allX,allY),batch_size=64,shuffle=True)
    for ep in range(20):
        model.train(); tot=0
        for X,y in warm_loader:
            opt.zero_grad(); out=model(X); loss=crit(out,y)
            loss.backward(); opt.step(); tot+=loss.item()
        sch.step()
    print("Warmup done ✓")

    evals, finals = [], []
    for tidx,(trainL,testL,sy) in enumerate(tasks):
        lam_SI = 0.05 + 0.15 * (1 - np.cos(np.pi * tidx / (num_tasks-1))) / 2
        lam_FIM = 0.05 + 0.25 * (1 - np.cos(np.pi * tidx / (num_tasks-1))) / 2
        print(f"Task {tidx+1}/{num_tasks}: λSI={lam_SI:.3f}, λFIM={lam_FIM:.3f}")
        for ep in range(200):
            model.train(); tot=0
            for X,y in trainL:
                opt.zero_grad(); out=model(X)
                loss=crit(out,y)
                if tidx>0:
                    loss += lam_SI*si.penalty()
                    if FIM_prev:
                        fim_loss=sum((FIM_prev[n]*(p-si.prev[n])**2).sum()
                                     for n,p in model.named_parameters() if p.requires_grad)
                        loss += lam_FIM*fim_loss
                loss.backward()
                for name,param in model.named_parameters():
                    if param.grad is not None:
                        param.grad = ema_decay * param.grad + (1-ema_decay) * param.grad.detach()
                si.accumulate()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); tot+=loss.item()
            sch.step()
            if (ep+1)%50==0:
                r2=evaluate(model,testL,sy)
                print(f"  Epoch {ep+1}/200 Loss={tot/len(trainL):.4f} R²={r2:.4f}")
        si.update(); FIM_prev=compute_FIM(model,trainL,crit)
        r2e=evaluate(model,testL,sy); evals.append(r2e)
        print(f"Task {tidx+1} Eval R²={r2e:.4f}")

    for i,(_,testL,sy) in enumerate(tasks):
        r2f=evaluate(model,testL,sy); finals.append(r2f)
        print(f"Final R² Task {i+1}: {r2f:.4f}")

    # ---- Metrics calculation (R² only) ----
    forgetting_r2 = [round(e - f, 4) for e, f in zip(evals, finals)]
    memory_stability_r2 = round(1 - np.mean(forgetting_r2), 5)

    print("========= R² PERFORMANCE MATRIX =========")
    print(f"Initial  R² per Task: {np.round(evals, 4)}")
    print(f"Final    R² per Task: {np.round(finals, 4)}")
    print(f"Forgetting (R²):     {forgetting_r2}")
    print(f"Memory Stability (R²): {memory_stability_r2}")
    print("=========================================")


# ========= Visualization Section =========
    tasks = np.arange(1, len(evals)+1)
    sns.set_style("whitegrid")

    # 1️⃣ Initial vs Final R² per task
    plt.figure(figsize=(10,5))
    plt.plot(tasks, evals, marker='o', label='Initial R²', linewidth=2)
    plt.plot(tasks, finals, marker='s', label='Final R²', linewidth=2)
    plt.title("Initial vs Final R² per Task", fontsize=14)
    plt.xlabel("Task Number", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2️⃣ Forgetting (R²) per task
    plt.figure(figsize=(10,5))
    plt.bar(tasks, forgetting_r2, color=sns.color_palette("crest", len(tasks)))
    plt.title("R² Forgetting per Task", fontsize=14)
    plt.xlabel("Task Number", fontsize=12)
    plt.ylabel("Forgetting (ΔR²)", fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3️⃣ Memory Stability summary
    plt.figure(figsize=(4,4))
    plt.bar(["Memory Stability (R²)"], [memory_stability_r2], color="mediumseagreen")
    plt.ylim(0, 1.05)
    plt.title("Overall Memory Stability (R²)", fontsize=13)
    plt.tight_layout()
    plt.show()

    return memory_stability_r2

if __name__=="__main__":
    ms = hybrid_train("/kaggle/working/CLM/Processed Data/Africa_mpox.csv", num_tasks=10)
    print(f"Final Memory Stability (Africa_mpox): {ms:.4f}")
