# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import mydlmodules


class EncoderLayer(torch.nn.Module):
    '''Encoder層

    Args:
        d_val (int): 入力値の次元数
        d_at (int): 注意(Attention)の次元数
        n_heads (int): 注意機構の数

    Attributes:
        at (mydlmodules.MultiHeadAttention): 多頭注意機構
        ff (mydlmodules.FeedForward): 全結合層2つ
        norm_01 (torch.nn.modules.normalization.LayerNorm): 'Multi-Head Attention'後の正規化
        norm_02 (torch.nn.modules.normalization.LayerNorm): 'Feed Forward'後の正規化
    '''

    def __init__(self, d_val, d_at, n_heads=1):
        super().__init__()
        self.at = mydlmodules.MultiHeadAttetion(d_val, d_at, n_heads)

        self.ff = mydlmodules.FeedForward(d_val, d_val)

        self.norm_01 = nn.LayerNorm(d_val)
        self.norm_02 = nn.LayerNorm(d_val)

    def forward(self, x):
        '''Encoderの順伝播

        Args:
            x (torch.Tensor): 入力値

        Returns:
            x (torch.Tensor): 出力値
        '''
        a = self.at(x)
        x = self.norm_01(x + a)

        a = self.ff(x)
        x = self.norm_02(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    '''Dncoder層

    Args:
        d_val (int): 入力値の次元数
        d_at (int): 注意(Attention)の次元数
        n_heads_01 (int): 'Self-Attention'の注意機構の数
        n_heads_02 (int): 'SourceTarget-Attention'の注意機構の数

    Attributes:
        at_01 (mydlmodules.MultiHeadAttention): 多頭注意機構
        at_02 (mydlmodules.MultiHeadAttention): Encoder層の出力をValueとKeyに利用した多頭注意機構
        ff (mydlmodules.FeedForward): 全結合層2つ
        norm_01 (torch.nn.modules.normalization.LayerNorm): at_01の'Multi-Head Attention'後の正規化
        norm_02 (torch.nn.modules.normalization.LayerNorm): at_02の'Multi-Head Attention'後の正規化
        norm_03 (torch.nn.modules.normalization.LayerNorm): 'Feed Forward'後の正規化
    '''

    def __init__(self, d_val, d_at, n_heads_01=1, n_heads_02=1):
        super().__init__()
        self.at_01 = mydlmodules.MultiHeadAttetion(d_val, d_at, n_heads_01)
        self.at_02 = mydlmodules.MultiHeadAttetion(d_val, d_at, n_heads_02)
        self.ff = mydlmodules.FeedForward(d_val, d_val)

        self.norm_01 = nn.LayerNorm(d_val)
        self.norm_02 = nn.LayerNorm(d_val)
        self.norm_03 = nn.LayerNorm(d_val)

    def forward(self, x, enc):
        '''Dncoderの順伝播

        Args:
            x (torch.Tensor): 入力値
            enc (torch.Tensor): Encoderの出力

        Returns:
            x (torch.Tensor): 出力値
        '''
        a = self.at_01(x)
        x = self.norm_01(a + x)

        a = self.at_02(x, en_out=enc)
        x = self.norm_02(a + x)

        a = self.ff(x)

        x = self.norm_03(x + a)
        return x


class TSTransformer(pl.LightningModule):
    '''TimeSeries_Transformer
        時系列分析(回帰)用Transformer

    Args:
        d_val (int): 入力値の次元数
        d_at (int): 注意(Attention)の次元数
        input_len (int): 入力値の長さ
        output_len (int): 出力値の長さ
        n_enc_layers (int): Encoder層の数
        n_dec_layers (int): Decoder層の数
        n_enc_heads (int): Encoder内の注意機構の数
        n_dec01_heads (int): Decoder内の注意機構の数
        n_dec02_heads (int): Decoder内の注意機構の数
        lr (float): 学習率

    Attributes:
        input_len (int): 入力値の長さ
        lr (float): 学習率
        pos (mydlmodules.PositionalEncoding): 位置エンコーディングを追加
        enc_layers (torch.nn.modules.container.ModuleList): Encoder層
        dec_layers (torch.nn.modules.container.ModuleList): Decoder層
        enc_input_fc (torch.nn.modules.linear.Linear): Encoder入力値の全結合層
        dec_input_fc (torch.nn.modules.linear.Linear): Decoder入力値の全結合層
        out_fc (torch.nn.modules.linear.Linear): 出力層入力値の全結合層
    '''

    def __init__(self, d_val, d_at, input_len, output_len, n_enc_layers=1, n_dec_layers=1, n_enc_heads=1, n_dec01_heads=1, n_dec02_heads=1, lr=0.01):
        super().__init__()

        self.input_len = input_len
        self.lr = lr

        self.pos = mydlmodules.PositionalEncoding(d_val)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_val, d_at, n_enc_heads) for _ in range(n_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_val, d_at, n_dec01_heads, n_dec02_heads) for _ in range(n_dec_layers)])

        self.enc_input_fc = nn.Linear(1, d_val)
        self.dec_input_fc = nn.Linear(1, d_val)
        self.out_fc = nn.Linear(input_len * d_val, output_len)

    def forward(self, x):
        '''Transformerの順伝播

        Args:
            x (torch.Tensor): 入力値

        Returns:
            x (torch.Tensor): 出力値
        '''
        # Encoder層
        e = self.enc_layers[0]
        e = e(self.pos(self.enc_input_fc(x)))
        for enc in self.enc_layers[1:]:
            e = enc(e)

        # Decoder層
        d = self.dec_layers[0](self.dec_input_fc(x[:, -self.input_len:]), e)
        for dec in self.dec_layers[1:]:
            d = dec(d, e)

        # Decoder層後の全結合層
        x = self.out_fc(d.flatten(start_dim=1))

        return x

    def training_step(self, batch, batch_idx):
        '''学習

        Args:
            batch (tuple): (入力値, 予測値)
            batch_idx (int): batchのindex

        Returns:
            loss (torch.Tensor): 損失関数の値
        '''
        x, t = batch
        y = self(x)
        criterion = nn.L1Loss()  # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
        loss = criterion(y, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        '''検証

        Args:
            batch (tuple): (入力値, 予測値)
            batch_idx (int): batchのindex

        Returns:
            loss (torch.Tensor): 損失関数の値
        '''
        x, t = batch
        y = self(x)
        criterion = nn.L1Loss()
        loss = criterion(y, t)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        '''テスト

        Args:
            batch (tuple): (入力値, 予測値)
            batch_idx (int): batchのindex

        Returns:
            loss (torch.Tensor): 損失関数の値
        '''
        x, t = batch
        y = self(x)
        criterion = nn.L1Loss()
        loss = criterion(y, t)

        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        '''最適化

        Args: None

        Returns:
            optimizer (torch.optim.Adam): '.step()'でパラメータを更新するオブジェクト
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class TSTransformerClass(TSTransformer):
    '''TimeSeries_Transformer
        時系列分析(分類)用Transformer
        引数も属性も'TSTransformer'と同じ
        メソッドの引数と返り値も'TSTransformer'と同じ
        分類問題用に変更した行にコメント付与
    '''

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        out_layer = nn.LogSoftmax(dim=1)  # softmaxで正規化
        out = out_layer(y)
        criterion = nn.NLLLoss()  # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html?highlight=nllloss#torch.nn.NLLLoss
        loss = criterion(out, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        out_label = torch.argmax(out, dim=1)
        acc = (out_label == t).sum() * 1.0 / len(t)  # モデルが出力した予測値と実際の予測値を用いて精度を算出
        self.log('train_acc', acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        out_layer = nn.LogSoftmax(dim=1)
        out = out_layer(y)
        criterion = nn.NLLLoss()
        loss = criterion(out, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        out_label = torch.argmax(out, dim=1)
        acc = (out_label == t).sum() * 1.0 / len(t)
        self.log('val_acc', acc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        out_layer = nn.LogSoftmax(dim=1)
        out = out_layer(y)
        criterion = nn.NLLLoss()
        loss = criterion(out, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        out_label = torch.argmax(out, dim=1)
        acc = (out_label == t).sum() * 1.0 / len(t)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss
