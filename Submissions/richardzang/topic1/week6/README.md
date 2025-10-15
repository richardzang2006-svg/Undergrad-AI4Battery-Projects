# Week6 · ESOL（Delaney）分子溶解度预测项目说明

本目录给出从数据获取、EDA、预处理、数据集划分、动态模型定义、训练与验证、测试与指标，到 I/O 批量预测的完整流程说明与示例代码。默认特征为 2048-bit Morgan 指纹（半径 2）。

> 建议配套查看本目录中的 Notebook（例如 `ESOL_new.ipynb`）。

---

## 1. 依赖环境（含 GPU 可选项）

- Python ≥ 3.10
- 核心：numpy, pandas, scikit-learn, torch（PyTorch）, rdkit
- 可选：matplotlib/seaborn（绘图）、tqdm（进度条）、joblib（缓存）、pyyaml（配置）

安装建议（rdkit 推荐使用 conda-forge 渠道，Windows 上更稳）：

- 使用 Conda：
    - 创建环境（示例）：
        - `conda create -n esol python=3.10`
    - 激活环境：
        - `conda activate esol`
    - 安装 rdkit（conda-forge）：
        - `conda install -c conda-forge rdkit`
    - 其余包（CPU 版本 PyTorch）：
        - `pip install numpy pandas scikit-learn torch matplotlib seaborn tqdm joblib pyyaml`

GPU（可选）：根据显卡与 CUDA 版本选对应的 PyTorch 轮子（参考 pytorch.org 指南），例如（示意，具体以官网为准）：

```powershell
# CUDA 12.x 例子（请以官网命令为准）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

> 小贴士：Notebook 开头打印版本信息，便于复现与定位问题。
>
> ```python
> import torch, sklearn, rdkit, numpy as np, pandas as pd
> print('torch', torch.__version__)
> print('cuda available:', torch.cuda.is_available())
> print('sklearn', sklearn.__version__)
> print('rdkit', rdkit.__version__)
> print('numpy', np.__version__, 'pandas', pd.__version__)
> ```

导出/保存环境（可复现）：

```powershell
conda env export -n esol > environment.yml
```

---

## 2. 数据获取（data.csv）

- 数据来源：DeepChem 提供的 Delaney（ESOL）处理后数据集。
- 如果本目录已有 `data.csv`，则可直接使用；若缺失，可用以下代码生成：

```python
# 生成 data.csv（官方顺序）
import pandas as pd
url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
df = pd.read_csv(url, skipinitialspace=True)[['smiles', 'measured log solubility in mols per litre']]
df.columns = ['smiles', 'logS']
df.to_csv('./data.csv', index=False)
print('data.csv ready, samples:', len(df))
```

> 说明：`smiles` 为分子 SMILES 字符串，`logS` 为对应的对数溶解度标签。

---

## 3. EDA（探索性数据分析）

目标：在不写代码的前提下明确数据质量与分布，形成分析清单：

- 列与类型：确认核心列 `smiles`（字符串）与 `logS`（连续值）。
- 完整性：检查缺失值与按 `smiles` 的重复记录，决定去重或聚合策略。
- 合法性：确认 SMILES 能被 RDKit 解析；无法解析的样本在特征阶段以零向量兜底并记录。
- 分布与极值：查看 `logS` 的分布形态与潜在离群点，避免训练被极端值主导。

---

## 4. 预处理与特征（2048-bit Morgan 指纹）

为保证训练/评估一致性，建议统一使用 RDKit 的 `rdFingerprintGenerator` 接口，并通过 `DataStructs.ConvertToNumpyArray` 转换为 `numpy` 数组。下面的函数包含了“无效 SMILES 兜底”“固定参数（radius=2, fpSize=2048）”“返回 float32”的细节：

```python
import numpy as np
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit import DataStructs

# 按固定参数创建指纹生成器（全局复用，避免重复构造带来的开销）
GEN_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def feat_array(smiles: str) -> np.ndarray:
    """将单个 SMILES 转换为 2048 维 Morgan 指纹（float32）。

    - 无效 SMILES 则返回全零向量，保持下游 pipeline 可运行；
    - 明确使用 fpSize=2048 与 radius=2，保证训练/评估一致；
    - 使用 DataStructs.ConvertToNumpyArray 做可靠转换。
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    bv = GEN_MORGAN.GetFingerprint(mol)  # ExplicitBitVect
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

# 批量生成并可缓存为 .npz，避免每次重复计算
import pandas as pd
smiles = pd.read_csv('./data.csv').smiles.values
X = np.vstack([feat_array(s) for s in smiles]).astype(np.float32)
np.savez_compressed('./features_morgan2048.npz', X=X)
```

> 可选：对标签 `logS` 做 z-score 标准化以稳定训练（预测后再反标准化）。

---

## 5. 数据集划分（顺序 8:1:1 + 可选方案）

- 基准方案（官方顺序 8:1:1）：
  - 训练：前 80%
  - 验证：中间 10%
  - 测试：最后 10%

```python
import numpy as np, pandas as pd

df = pd.read_csv('./data.csv')
y = df.logS.values.astype(np.float32)  # 标签向量
X = np.load('./features_morgan2048.npz')['X']  # 2048 维特征

# 官方顺序切分：前 80% 训练，中间 10% 验证，最后 10% 测试
n = len(X)
train_cut = int(0.8 * n)
val_cut   = int(0.9 * n)
X_train, y_train = X[:train_cut], y[:train_cut]
X_val,   y_val   = X[train_cut:val_cut], y[train_cut:val_cut]
X_test,  y_test  = X[val_cut:], y[val_cut:]
```

- 可选方案：随机划分或“分子骨架（scaffold）划分”，用于更严格的泛化评估（实现更复杂，此处不展开）。

---

## 6. 动态模型定义（以 MLP 为例，可配置层宽/Dropout）

```python
import torch, torch.nn as nn

class MLP(nn.Module):
    """简单的多层感知机回归模型。

    参数：
    - n_in: 输入维度，默认为 2048（Morgan 指纹位数）
    - hidden: 第一层隐藏单元数
    - drop: Dropout 比例，缓解过拟合
    - act: 激活函数类型（nn.ReLU 默认，可替换为 GELU/SiLU 等）
    - widths: 额外层宽配置（list），允许更灵活的网络深度
    """
    def __init__(self, n_in=2048, hidden=512, drop=0.3, act=nn.ReLU, widths=(256, 128)):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), act(), nn.Dropout(drop)]
        in_dim = hidden
        for w in widths:
            layers += [nn.Linear(in_dim, w), act(), nn.Dropout(drop)]
            in_dim = w
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 2048] -> [B]
        return self.net(x).squeeze(1)

# 可配置工厂函数（便于通过 dict/yaml 传参）
def build_model(cfg):
    return MLP(n_in=cfg.get('n_in', 2048),
               hidden=cfg.get('hidden', 512),
               drop=cfg.get('drop', 0.3),
               act=cfg.get('act', nn.ReLU),
               widths=cfg.get('widths', (256, 128)))
```

> 若需命令行/Notebook 参数化，可把 `cfg` 做成字典或用 `argparse`/`yaml` 配置。

---

## 7. 训练与验证（早停、调度器、元信息保存）

```python
import json, time
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cfg = dict(n_in=2048, hidden=512, drop=0.3, widths=(256,128))
model = build_model(model_cfg).to(device)

# 构建 DataLoader（训练集 shuffle 打散）
batch = 64
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, pin_memory=torch.cuda.is_available())

# 将验证集预张量化，避免每个 epoch 重复构造
val_X = torch.tensor(X_val).to(device)
val_y = torch.tensor(y_val).to(device)

# 优化器与（可选）学习率调度器
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=True)

# 训练循环（含早停与最佳模型保存）
best = float('inf'); patience = 50
history = []
for epoch in range(500):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = nn.MSELoss()(preds, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(train_ds)

    # 验证 RMSE
    model.eval()
    with torch.no_grad():
        val_rmse = torch.sqrt(nn.MSELoss()(model(val_X), val_y)).item()
    sched.step(val_rmse)  # 调度器根据指标调整学习率

    history.append(dict(epoch=epoch, train_mse=epoch_loss, val_rmse=val_rmse, lr=opt.param_groups[0]['lr']))

    # 维护最佳模型
    if val_rmse < best - 1e-4:
        best, patience = val_rmse, 50
        torch.save(model.state_dict(), './best_esol.pt')
        # 同步保存元信息（便于复现）
        meta = dict(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            best_val_rmse=best,
            model_cfg=model_cfg,
            optimizer='AdamW', lr=3e-4, weight_decay=1e-4,
            scheduler='ReduceLROnPlateau', batch=batch,
        )
        with open('./esol_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    else:
        patience -= 1
        if patience == 0:
            print('Early stop'); break
    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d} | train MSE {epoch_loss:.4f} | val RMSE {val_rmse:.3f} | lr {opt.param_groups[0]['lr']:.2e}')
```

> 复现性：固定随机种子、关闭 CUDNN 不确定性。
>
> ```python
> import torch, numpy as np, random, os
> SEED = 42
> random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
> if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
> torch.backends.cudnn.deterministic = True
> torch.backends.cudnn.benchmark = False
> os.environ['PYTHONHASHSEED'] = str(SEED)
> ```

---

## 8. 测试与指标（批量推理，兼容权重加载）

- 使用测试集（最后 10%）评估并报告 RMSE、R²。
- 建议批量化推理，避免一次性占用过多显存。

```python
import numpy as np, torch
from sklearn.metrics import root_mean_squared_error, r2_score

def predict_in_batches(model, X, device, bs=1024):
    """对特征矩阵 X 做批量推理，节省显存与内存占用。"""
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)

# 加载最优权重（兼容老版本 PyTorch）
model = build_model({}).to(device)
try:
    state = torch.load('./best_esol.pt', map_location=device, weights_only=True)
except TypeError:
    state = torch.load('./best_esol.pt', map_location=device)
model.load_state_dict(state)

# 推理与指标
y_pred = predict_in_batches(model, X_test, device)
print('Test RMSE:', root_mean_squared_error(y_test, y_pred))
print('Test R²  :', r2_score(y_test, y_pred))
```

---

## 9. 本模型 RMSE 的“极限”说明（经验范围）

- 在“2048-bit Morgan 指纹 + 简单 MLP + 官方顺序 8:1:1 划分”的设定下，常见的 RMSE 经验范围大致在 0.6~0.8（logS）。
- 实际值受：随机种子、指纹实现细节（位向量生成方式与转换）、超参数（隐藏层/Dropout/LR/正则）、是否做标签标准化、是否使用早停/学习率调度等影响。
- 更复杂的模型（如图神经网络、骨架划分）或不同的数据预处理可能得到不同（更低/更高）的 RMSE；因此该范围仅作经验参考，不构成硬性下界。

---

## 10. I/O 批量预测（从 SMILES/文本到 CSV）

示例：从 `predict_input.csv`（含一列 `smiles`）读取，输出 `predict_output.csv`（含 `smiles,pred_logS`）。

```python
import pandas as pd, numpy as np, torch
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit import DataStructs

# 统一指纹函数（与训练一致）
GEN_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def feat_array(smiles: str) -> np.ndarray:
    """预测阶段的特征函数：无效 SMILES 返回零向量并保留顺序。"""
    mol = MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    bv = GEN_MORGAN.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

# 读取待预测 SMILES（CSV 必含列 `smiles`；若为 TXT 可先读为列表再构建 DataFrame）
df_in = pd.read_csv('./predict_input.csv')
X_pred = np.vstack([feat_array(s) for s in df_in.smiles.values]).astype(np.float32)

# 加载模型并预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(n_in=2048).to(device)
try:
    state = torch.load('./best_esol.pt', map_location=device, weights_only=True)
except TypeError:
    state = torch.load('./best_esol.pt', map_location=device)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    y_hat = model(torch.tensor(X_pred).to(device)).cpu().numpy()

pd.DataFrame({'smiles': df_in.smiles, 'pred_logS': y_hat.reshape(-1)}).to_csv('./predict_output.csv', index=False)
print('Saved to ./predict_output.csv')
```

---

## 11. 目录结构建议

```text
week6/
├─ data.csv                     # 数据（若缺失可用脚本生成）
├─ features_morgan2048.npz      # 可选：缓存的特征
├─ best_esol.pt                 # 训练出的最优权重
├─ ESOL_new.ipynb               # 训练/评估 Notebook
├─ README.md                    # 本说明
└─ predict_input.csv            # 可选：待预测 SMILES
```

---

## 12. 复现步骤（建议）

1) 准备环境与依赖（确保 `rdkit` 安装成功）。
2) 生成 `data.csv`（若不存在）。
3) 统一指纹函数，批量生成 `X`，可缓存至 `features_morgan2048.npz`。
4) 官方顺序 8:1:1 划分，构建 `X_train/X_val/X_test` 与 `y`。
5) 训练（监控 val RMSE，早停，保存 `best_esol.pt`）。
6) 加载最优模型在测试集评估 RMSE/R²。
7) （可选）对外部 SMILES 进行批量预测，生成 `predict_output.csv`。

---

## 13. 常见问题排查

- RDKit 安装失败：优先使用 `conda install -c conda-forge rdkit`；Windows 上不建议用 `pip` 安装。
- SMILES 无法解析：跳过或以零向量代替，并在日志中记录；必要时清洗输入。
- 指纹维度不一致：确保训练与预测使用完全相同的 `radius=2, fpSize=2048` 与相同的转换路径。
- `torch.load(..., weights_only=True)` 报 `TypeError`：说明 PyTorch 版本较旧，去掉 `weights_only=True` 重新加载。
- 指标不稳定：固定随机种子、启用早停、考虑对标签做标准化并反标准化评估。
- CUDA 相关：`torch.cuda.is_available()` 为 False 时表示当前环境/驱动/安装包不支持 GPU，使用 CPU 模式或重新按官网指令安装对应 CUDA 版本的 PyTorch。
- 显存不足：降低 batch size，或使用批量推理函数；训练时可裁小模型宽度/深度或提高 Dropout。
- R² 为负：表示模型劣于常数基线，检查特征/标签对齐、指纹一致性、是否误用测试集做训练等。

---

如需进一步自动化（命令行脚本/配置文件/多次重复试验统计），可在本 README 的基础上拆分出 `utils.py / train.py / eval.py / predict.py` 并引入 `argparse/yaml` 做统一管理。
