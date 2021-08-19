# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import math
import os


class PositionalEncoding(nn.Module):
    '''位置エンコーディングの生成
    下記URLの実装から、'Dropout'を消去したコード
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        d_model (int): 入力値の次元数。sinとcosの2種で位置情報を付加しているので、d_modelが奇数だとerror('pe[:, 0::2]'と'pe[:, 1::2]'で2次元目の次元数が変わってしまう)。
        max_len (int): 位置エンコーディングの長さ

    Attributes:
        register_buffer (torch.jit.ScriptModule.register_buffer): モジュールに追加されるBuffer (https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=register_buffer#torch.jit.ScriptModule.register_buffer)

    '''

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, input):
        '''位置エンコーディングの順伝播

        Args:
            input (torch.Tensor): 入力値

        Returns:
            output (torch.Tensor): 位置情報を付加した入力値

        '''
        output = input + self.pe[:input.size(1), :]. squeeze(1)
        return output


def attention(q, k):
    '''"Attention　Weight"の算出と正規化
        縮小付きの内積注意(Scaled Dot-Product Attention)
        半精度学習による推論高速化を行うためには、この関数内で32bitデータ型への変換を行わないこと

        Args:
            q (torch.Tensor): Query
            k (torch.Tensor): Key

        Returns:
            norm_weight (torch.Tensor): softmaxで正規化された'Attention Weight' = Queryと各Keyの内積を'類似度'とし、softmaxで正規化することでQueryに一致したKeyを表す
    '''

    a_weight = torch.matmul(q, k.transpose(2, 1)) / torch.sqrt(torch.tensor(q.shape[-1]))
    norm_weight = F.softmax(a_weight, -1)

    return norm_weight


class Attetion(torch.nn.Module):
    '''注意機構
        入力値から'Attention'を計算し、対応するValueとの積を算出

    Args:
        d_val (int): 入力値の次元数
        d_at (int): 注意(attention)の次元数

    Attributes:
        query (torch.Tensor): Query
        key (torch.Tensor): Key
        value (torch.Tensor): Value
    '''

    def __init__(self, d_val, d_at):
        super().__init__()
        self.value = nn.Linear(d_val, d_val, bias=False)
        self.key = nn.Linear(d_val, d_at, bias=False)
        self.query = nn.Linear(d_val, d_at, bias=False)

    def forward(self, x, en_out=None):
        '''Attentionの順伝播

        Args:
            x (torch.Tensor):入力値
            en_out (torch.Tensor): decoderに入力されるencoderの計算結果

        Returns:
            output (torch.Tensor): 'Attention'に対応したValue

        '''

        if en_out == None:
            # Encoder
            a_weights = attention(self.query(x), self.key(x))
            output = torch.matmul(a_weights, self.value(x))

            return output

        else:
            # Decoder
            a_weights = attention(self.query(x), self.key(en_out))
            output = torch.matmul(a_weights, self.value(en_out))

            return output


class MultiHeadAttetion(torch.nn.Module):
    '''多頭注意機構
        注意機構を複数連結させたもの

    Args:
        d_val (int): 入力値の次元数
        d_at (int): 注意(Attention)の次元数
        n_heads (int): 注意機構の数

    Attributes:
        heads (torch.nn.modules.container.ModuleList): 全ての注意機構
        fc (torch.nn.modules.linear.Linear): 出力層直前の全結合層
    '''

    def __init__(self, d_val, d_at, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Attetion(d_val, d_at) for _ in range(n_heads)])
        self.fc = nn.Linear(n_heads * d_val, d_val, bias=False)

    def forward(self, x, en_out=None):
        '''MultiHeadAttentionの順伝播

        Args:
            x (torch.Tensor):入力値
            en_out (torch.Tensor): decoderで利用されるencoderの出力

        Returns:
            output (torch.Tensor): 出力層直前の値
        '''
        attention_out = [head(x, en_out=en_out) for head in self.heads]

        multi_attention = torch.stack(attention_out, dim=-1)
        multi_attention = multi_attention.flatten(start_dim=2)

        output = self.fc(multi_attention)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_val, d_ff=1024, dropout=0.0):
        '''全結合層2つで入力を変換
        
        Args:
            d_val (int): 入力値の次元数
            d_ff (int): 第1層出力値の次元数
            dropout (float): 入力を'0'にする割合。過学習対策。
            
        Attributes:
            fc_01 (torch.nn.modules.linear.Linear): 全結合層
            dropout (torch.nn.Dropout): 入力を'0'にする
            fc_02 (torch.nn.modules.linear.Linear): 全結合層
            
        '''
        super().__init__()

        self.fc_01 = nn.Linear(d_val, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc_02 = nn.Linear(d_ff, d_val)

    def forward(self, x):
        '''

        Args:
            x (torch.Tensor): 入力

        Returns:
            x (torch.Tensor): 出力

        '''
        
        x = self.fc_01(x)
        x = self.dropout(F.relu(x))
        x = self.fc_02(x)

        return x


def tde_generator_regression(input_series, time_diff, batch_size, shuffle_flag):
    '''回帰タスク用TDE(Time Delay Embedding)のdataloaderを作成
       TDEの詳細は下記URl参照
       https://blog.brainpad.co.jp/entry/2021/02/17/140000

    Args:
        input_series (pandas.core.series.Series): 単変量時系列データ
        time_diff_list (list): 時間間隔
        batch_size (int): バッチサイズ
        shuffle_flag (bool): サンプルをランダムに抽出するかどうか

    Returns:
       tde_dataloader (torch.utils.data.dataloader.DataLoader): 時間間隔で過去の値を連結したlen(time_diff_list)次元のベクトルと、予測対象となる値をまとめたDataloader

    '''

    x = []
    t = []

    for i in range(len(input_series) - time_diff - 1):
        x.append(input_series[i:i + time_diff])
        t.append(input_series[i + time_diff])
    x = torch.Tensor(x).unsqueeze(-1)  # transformerの入力に必要な次元の確保
    t = torch.Tensor(t).unsqueeze(-1)

    tde_dataset = TensorDataset(x, t)
    tde_dataloader = DataLoader(tde_dataset, batch_size, shuffle=shuffle_flag, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)

    return tde_dataloader


def tde_generator_class(input_series, label_series, time_diff_list, batch_size, shuffle_flag):
    '''分類タスク用TDE(Time Delay Embedding)のdataloaderを作成
       TDEの詳細は下記URl参照
       https://blog.brainpad.co.jp/entry/2021/02/17/140000

    Args:
        input_series (pandas.core.series.Series): 単変量時系列データ
        label_series (pandas.core.series.Series): 予測対象となる分類クラスデータ
        time_diff_list (list): 時間間隔
        batch_size (int): バッチサイズ
        shuffle_flag (bool): サンプルをランダムに抽出するかどうか

    Returns:
       tde_dataloader (torch.utils.data.dataloader.DataLoader): 時間間隔で過去の値を連結したlen(time_diff_list)次元のベクトルと、予測対象となるクラスをまとめたDataloader

    '''

    x = []
    t = []

    for obj_index in range(time_diff_list[len(time_diff_list) - 1], len(input_series) - 1):
        mini_x = []
        for back_start in time_diff_list[::-1]:
            mini_x.append(input_series.iloc[obj_index - back_start])
        x.append(mini_x)
        t.append(label_series.iloc[obj_index])

    x = torch.Tensor(x).unsqueeze(-1)  # transformerの入力に必要な次元の確保。2次元以上のlistはtorch.Tensor()で変換しないとエラー
    t = torch.tensor(t, dtype=torch.long)

    tde_dataset = TensorDataset(x, t)
    tde_dataloader = DataLoader(tde_dataset, batch_size, shuffle=shuffle_flag, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)

    return tde_dataloader
