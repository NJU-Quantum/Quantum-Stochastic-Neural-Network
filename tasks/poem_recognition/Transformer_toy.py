# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:46:27 2025

@author: 12704
"""

import random
import textwrap
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tokenizer import SimpleTokenizer
from data import data_circuit_str, data_input_density_matrix, data_output_density_matrix, data_input_realvec, data_output_realvec, realvec_to_hermitian, hermitian_to_realvec

random.seed(13)
torch.manual_seed(13)
file_name = "book1.txt" #训练集+验证集

#超参数
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu" #选择cpu还是gpu

block_size = 11 #训练、验证的字符串长度
batch_size = 5 #平行处理的独立分组数,可以理解为并行计算

embedding_dim = 64 #将token嵌入到的向量空间的维数
num_heads = 1 #注意力头个数
head_size = embedding_dim // num_heads #注意力头大小
n_layer = 4 #嵌套层数

learning_rate = 0.0005 #学习率,只影响优化过程
max_iters = 1000 #循环次数
eval_interval= max_iters//10 #验证间隔
eval_iters = 100 #单次验证集大小

dropout = 0.1 #Dropout通过在训练神经网络期间随机丢弃单元来防止过拟合

# 数据预处理
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.split('\n') #list, 每个元素是一行字符串
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(text)

text = ' '.join(text) #合并为一个字符串
#text = text.split(' ')
#print(text)
# 创建字典
chars = tokenizer.word2id
vocab_size = len(chars) #字典大小,即一共多少个不同的字
print(chars)
print(f"字典长度:{vocab_size}")

# 字符和token互转,token的数据类型是int
encode = lambda str1: tokenizer.encode(str1) #把str转化为token的list
decode = lambda list1: tokenizer.decode(list1) #把list转为str

# 数据分组
data1 = torch.tensor(encode(text)) #1维tensor
n_train = int(0.75*len(data1))
data_train = data1[:n_train]
data_val = data1[n_train:]
print(f"文件{file_name}读取完成")  

# 数据分组 for QC
data2 = torch.tensor([encode(i) for i in data_circuit_str]) 
data_input_realvec_tensor = torch.tensor(data_input_realvec, dtype=torch.float32)
data_output_realvec_tensor = torch.tensor(data_output_realvec, dtype=torch.float32)
# 新增:实数向量维度(必须在 LanguageModel 使用 realvec_dim 之前定义)
realvec_dim = data_input_realvec_tensor.shape[1]

def get_batch(split):
    ix = torch.randint(len(split) - block_size, (batch_size,))
    #ix = ix // 5 * 5  # 保证起始位置是5的倍数
    x = torch.stack([split[i:i+block_size] for i in ix]) # 输入值batch, stack为拼接函数
    y = torch.stack([split[i+1:i+block_size+1] for i in ix]) # 把x向后移一个字符,想要的输出值target
    x, y = x.to(device), y.to(device) #x.shape = batch_size, block_size
    return x, y

def get_batch_qc_train():
    ix = torch.randint(int(0.75*len(data2)), (batch_size,))
    x0_realvec = torch.stack([data_input_realvec_tensor[i] for i in ix])
    x = torch.stack([data2[i] for i in ix]) # 输入值batch, stack为拼接函数
    y_realvec = torch.stack([data_output_realvec_tensor[i] for i in ix])
    x0_realvec, x, y_realvec = x0_realvec.to(device), x.to(device), y_realvec.to(device)
    return x0_realvec, x, y_realvec

def get_batch_qc_val():
    ix = torch.randint(int(0.25*len(data2)) + int(0.75*len(data2)), (batch_size,))  # 可选:从验证区间抽取,或直接用不同种子
    # 这里保持随机抽样到整个集合也可,根据需求调整
    ix = torch.randint(int(0.75*len(data2)), (batch_size,))
    x0_realvec = torch.stack([data_input_realvec_tensor[i] for i in ix])
    x = torch.stack([data2[i] for i in ix])
    y_realvec = torch.stack([data_output_realvec_tensor[i] for i in ix])
    x0_realvec, x, y_realvec = x0_realvec.to(device), x.to(device), y_realvec.to(device)
    return x0_realvec, x, y_realvec

print(torch.__version__)
print(torch.cuda.is_available())
#print(get_batch(data_train))
#def one_hot_posi_embedding(block_size, embedding_dim):


#--损失评测----
@torch.no_grad()#不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model):
    out = {}
    model.eval()#把模型转化为evaluate模式(默认模式是train)
    losses=torch.zeros(eval_iters)#建立一个初始值为0的容器,用于储存loss值
     # 如果模型有 predict_real,按 QC 模式评估(MSE)；否则保留原 token 交叉熵评估
    if hasattr(model, 'predict_real'):
        for k in range(eval_iters):
            x0, X, y_real = get_batch_qc_train()
            pred, loss = model(X, x0_realvec=x0, y_realvec=y_real)
            losses[k] = loss.item()
        out['train'] = losses.mean()
        for k in range(eval_iters):
            x0, X, y_real = get_batch_qc_val()
            pred, loss = model(X, x0_realvec=x0, y_realvec=y_real)
            losses[k] = loss.item()
        out['val'] = losses.mean()
    else:
        for k in range(eval_iters):
            X, Y = get_batch(data_train)#split是一个字符串,用来控制get_batch()函数的行为

            logits, loss = model(X, Y) #model的输入值一个是index(以每个字符的序号表示的序列),一个是target
            losses[k] = loss.item()
            out['train'] = losses.mean()#out是含有两个元素的字典,一个是train,一个是val,每个元素对应一个loss的平均值
        for k in range(eval_iters):
            X, Y = get_batch(data_val)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out['val'] = losses.mean()
    model.train()#再转化为训练模式(模型建立后默认为训练模式)
    return out


# Head类
class Head(nn.Module):
    def __init__(self, head_size):#比如只有一个head,head_size = embedding_dim
        super().__init__()
        
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)# 线性变换层
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))#不可训练的,结构(约等于常量)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #B, T, C = x.shape, B = batch_size, T = block_size, C = embedding_dim
        k = self.key(x) # (B, T, C) -> (B, T, head_size)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**(-0.5)  #注意力方阵(B, T, T) #3维数组的@,最后两个维度做矩阵乘法
        wei = wei.masked_fill(self.tril == 0, float("-inf")) #掩码填充 masked fill(mask, value)-> Tensor
        wei = F.softmax(wei, dim=-1) #对最后一个维度(行向量)做softmax
        wei = self.dropout(wei) #随机去掉(归零)一些值,增加网络的稳定性

        v = self.value(x) # v.shape = (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #拼接每个head的输出
        out = self.dropout(self.proj(out))
        return out

# 前馈层
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*4),
            nn.ReLU(),
            nn.Linear(embedding_dim*4, embedding_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Transformer块
class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, head_size):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size) # 多头注意力
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # 残差多头注意力网络
        x = x + self.ffwd(self.ln2(x)) # 残差线性前馈层
        return x

# LLM
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        """
            Hermitian 4x4 矩阵编码为长度 16 的实数向量
            顺序: [diag0, diag1, diag2, diag3,
           Re(0,1), Im(0,1),
           Re(0,2), Im(0,2),
           Re(0,3), Im(0,3),
           Re(1,2), Im(1,2),
           Re(1,3), Im(1,3),
           Re(2,3), Im(2,3)]
        """
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim) # conception space
        # 为了支持在序列首插入 x0_emb,把位置 embedding 长度扩展 1
        self.position_embedding_table = nn.Embedding(block_size + 1, embedding_dim) # position space
        #self.encoder_blocks = nn.Sequential(*[Block(embedding_dim, num_heads, head_size) for _ in range(n_layer)])
        self.encoder_blocks = nn.ModuleList([Block(embedding_dim, num_heads, head_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_dim) # final layer norm
        # QC-specific:从 realvec 到 embedding 的投影；以及从 pooled embedding 到 realvec 的预测头
        self.real_proj = nn.Linear(realvec_dim, embedding_dim)
        self.predict_real = nn.Linear(embedding_dim, realvec_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)


    def forward(self, idx, x0_realvec=None, targets=None, y_realvec=None):
        B, T = idx.shape # B = batch_size, T = block_size, 数据为token(整数)形式
        
        token_embd = self.token_embedding_table(idx)  # (B, T, E)
        # 如果提供 x0_realvec:把它投影为 embedding,并插入为序列第一项 -> (B, T+1, E)
        if x0_realvec is not None:
            x0_emb = self.real_proj(x0_realvec)                      # (B, E)
            # 拼接 x0_emb 到 token embeddings 的第一行
            x = torch.cat([x0_emb.unsqueeze(1), token_embd], dim=1)  # (B, T+1, E)
            pos_len = T + 1
            position_idx = torch.arange(pos_len, device=device)     # (T+1,)
            position_embd = self.position_embedding_table(position_idx).unsqueeze(0)  # (1, T+1, E)
            x = x + position_embd                                    # 广播到 (B, T+1, E)
        else:
            # 常规模式,不插入 x0
            position_idx = torch.arange(T, device=device)
            position_embd = self.position_embedding_table(position_idx)
            x = token_embd + position_embd # (B, T, embd_dim)
        for block in self.encoder_blocks: #x = self.encoder_blocks(x)
            x = block(x)
        x = self.ln_f(x)
        
        # QC 分支:pooling -> 预测 realvector(此处 pooling 包含插入的 x0 token)
        if x0_realvec is not None:
            pooled = x.mean(dim=1)  # (B, E)
            y_pred = self.predict_real(pooled)  # (B, realvec_dim)
            if y_realvec is None:
                loss = None
            else:
                loss = F.mse_loss(y_pred, y_realvec)
            return y_pred, loss
        
        # 原 token 分支(language modeling)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B2, T2, C = logits.shape
            logits = logits.reshape(B2*T2,C) #摊平, B*T个概率分布向量
            targets = targets.view(B2*T2) # B*T个index
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, x0_realvec=None, token_sequ=None, max_new_tokens=0): #token_sequ已知的上文,max_new_tokens是续写的长度(B,T)
        
        if token_sequ is None:
            raise ValueError("token_sequ must be provided")

        # 确保在正确设备上
        token_sequ = token_sequ.to(next(self.parameters()).device)

        # 如果需要先进行 token autoregressive 续写(保留原行为)
        if max_new_tokens > 0 and x0_realvec is None:
            # 原 LM 续写路径(不带 x0)
            for _ in range(max_new_tokens):
                tokens_input = token_sequ[:, -block_size:]
                logits, loss = self.forward(tokens_input) # logits,(B, T, vocab_size)
                logits = logits[:, -1, :]#只取字符串最后一个,概率分布向量格式
                probs = F.softmax(logits, dim=-1)
                token_next = torch.multinomial(probs, num_samples=1)#概率分布向量-->one-hot向量-->整数token
                token_sequ = torch.cat((token_sequ, token_next), dim=1)
            new_tokens = token_sequ[:, -max_new_tokens:]
            return new_tokens

        # QC 模式:有 x0_realvec -> 直接调用 forward 得到 y_pred
        if x0_realvec is not None:
            x0_realvec = x0_realvec.to(next(self.parameters()).device)
            # 截断 token_sequ 保证长度不超过 block_size(forward 会把 x0 插入为第一项)
            tokens_input = token_sequ[:, -block_size:]
            y_pred, _ = self.forward(tokens_input, x0_realvec=x0_realvec)
            return y_pred
    
        # 若没有 x0_realvec 且 max_new_tokens == 0,则返回最后一个 token 的 logits（与旧行为兼容）
        tokens_input = token_sequ[:, -block_size:]
        logits, _ = self.forward(tokens_input)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        token_next = torch.multinomial(probs, num_samples=1)
        return token_next




