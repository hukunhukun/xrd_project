# Transformer

## 1. input feature

### input data

- raw data：[batch_size, seq_len]
- token embedding：经过 embedding 处理后的数据：[batch_size, seq_len, embedding_dim] 使用 `nn.embedding()` 函数实现

```python
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
```

- position embedding：添加序列中的位置信息

```python
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
```

- embedding = position embedding + token embedding
- embeddding 之后的数据shape [batch_size,seq_len,embedding]

## 2. attention

### attention layer:

- 定义了Q, K, V：其中Q K V均为input，即 shape均为[batch_size,seq_len,embedding]
- 求出token之间的相似系数$Q\times K^T$ ，注意力系数矩阵attention：[batch_size, seq_len,seq_len]
- ${\rm output} = attention \times V$  [batch_size, seq_len,embedding]

```python
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
```

### multihead attention

- 这样下来并没有多少参数训练，提出多头注意力机制
- 将embedding维度分为几个部分，如果 input为[128,100,512]，embedding为512，分为8个头，则分为8个[128,100,64]的部分
- 使用nn.Linear() 函数进行切片，增加了参数
- 最后将这八个头concat
- output为 [batch_size,seq_len,embedding]

```python
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
```

## 3. output