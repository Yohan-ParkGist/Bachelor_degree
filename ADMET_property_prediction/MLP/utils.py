import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score, average_precision_score
import os, joblib

# import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
np.random.seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 고정된 랜덤 시드를 사용하여 재현 가능한 셔플링 설정
g = torch.Generator()
g.manual_seed(777)  # 고정된 시드 설정

if torch.cuda.is_available():
    torch.cuda.manual_seed(777)
    torch.cuda.manual_seed_all(777)  # 멀티 GPU 환경 시 사용
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Early Stopping을 위한 클래스 정의
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience  # 개선되지 않는 에포크를 기다릴 수 있는 횟수
        self.delta = delta  # 개선 기준이 되는 최소 변화량
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """검증 손실이 개선되었을 때 호출"""
        self.best_loss = val_loss
        
# Outlier 제거 함수
def remove_outliers(data, column, threshold):
    data[column] = data[column].apply(lambda x: np.nan if x > threshold else x)
    return data

# Fingerprint 불러오기 함수
def load_fingerprints(nbits, radius, file_fingerprint):
    file = os.path.join(file_fingerprint, f"fingerprints_{nbits}_{radius}.joblib")
    if os.path.exists(file):
        print(f"Loading fingerprints from {file}")
        return joblib.load(file)
    else:
        raise FileNotFoundError(f"Fingerprint file for nBits={nbits}, radius={radius} not found.")

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = len(self.X)  # 데이터의 길이 저장
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            idx = idx % self.length  # 인덱스가 데이터의 범위를 벗어나면 나머지 연산을 통해 인덱스를 조정
        return self.X[idx], self.y[idx]
    
class MLP(nn.Module):
    def __init__(self, nBits, drop_rate, seed=777):
        super(MLP, self).__init__()

        # 전역 랜덤 시드 고정
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 모델 레이어 정의
        self.fc1 = nn.Linear(nBits, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(drop_rate)

        # 가중치 초기화
        self._initialize_weights(seed)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = nn.Sigmoid()(self.fc3(x))
        return x

    def _initialize_weights(self, seed):
        torch.manual_seed(seed)  # 시드 고정

        # Fully connected layers
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, a=0, mode='fan_in', nonlinearity='linear')

        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)
        if self.fc3.bias is not None:
            nn.init.constant_(self.fc3.bias, 0)

        # BatchNorm layers
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)