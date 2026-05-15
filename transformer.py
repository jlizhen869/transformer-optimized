import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(TokenAndPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        token_embeddings = self.token_embed(x) * math.sqrt(self.d_model)
        return token_embeddings + self.pe[:, :seq_len]


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    # standard dot-product attention, optional attn dropout
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)  # 覆盖原变量
    return torch.matmul(attention_weights, V), attention_weights


class MultiHeadAttention(nn.Module):
    # fused W_qkv for both self-attention and cross-attention
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, num_splits=1):
        batch_size, seq_len, total_dim = x.size()
        if num_splits > 1:
            return x.view(
                batch_size, seq_len, num_splits, self.num_heads, self.d_k
            ).permute(2, 0, 3, 1, 4)
        else:
            return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, context=None, mask=None, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        if context is None:
            # self-attention
            qkv = self.split_heads(self.W_qkv(x), num_splits=3)
            cur_Q, cur_K, cur_V = qkv[0], qkv[1], qkv[2]
            if past_key_value is not None:
                prev_K, prev_V = past_key_value
                cur_K = torch.cat((prev_K, cur_K), dim=-2)
                cur_V = torch.cat((prev_V, cur_V), dim=-2)
        else:
            # cross-attention
            cur_Q = self.split_heads(
                self.W_qkv(x)[:, :, :self.d_model],
                num_splits=1
            )
            kv = self.W_qkv(context)[:, :, self.d_model:]
            kv_split = self.split_heads(kv, num_splits=2)
            cur_K, cur_V = kv_split[0], kv_split[1]

        present_key_value = (cur_K, cur_V) if use_cache else None
        attn_out, _ = scaled_dot_product_attention(cur_Q, cur_K, cur_V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_out), present_key_value


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layernorm1(x)
        attn_out, _ = self.mha(x, mask=mask)
        x = residual + self.dropout1(attn_out)

        residual = x
        x = self.layernorm2(x)
        x = residual + self.dropout2(self.ffn(x))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        causal_mask,
        src_padding_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = x
        x = self.layernorm1(x)
        attn_out, next_cache = self.masked_mha(
            x, mask=causal_mask, past_key_value=past_key_value, use_cache=use_cache
        )
        x = residual + self.dropout1(attn_out)

        residual = x
        x = self.layernorm2(x)
        attn_out, _ = self.cross_mha(x, context=encoder_output, mask=src_padding_mask)
        x = residual + self.dropout2(attn_out)

        residual = x
        x = self.layernorm3(x)
        x = residual + self.dropout3(self.ffn(x))

        return x, next_cache


class Transformer(nn.Module):
    """
    Encoder-decoder Transformer for sequence-to-sequence tasks (e.g. translation).
    Uses pre-LN, fused QKV projection, and KV cache for autoregressive decoding.

    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        d_model: model hidden dimension
        num_heads: number of attention heads (d_model must be divisible by num_heads)
        num_layers: number of encoder and decoder layers
        d_ff: feedforward inner dimension (typically 4 * d_model)
        max_seq_len: maximum sequence length
        dropout: dropout rate
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_len,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.src_embedding = TokenAndPositionalEmbedding(
            src_vocab_size, d_model, max_seq_len
        )
        self.tgt_embedding = TokenAndPositionalEmbedding(
            tgt_vocab_size, d_model, max_seq_len
        )
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        x = self.dropout(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, causal_mask, tgt_mask):
        x = self.dropout(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x, _ = layer(
                x, memory, causal_mask, tgt_mask, past_key_value=None, use_cache=False
            )
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        causal_mask = self._make_causal_mask(tgt, tgt_mask)
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, causal_mask, src_mask)
        return self.final_linear(output)

    @torch.no_grad()
    def generate(
        self, src, src_mask=None, max_len=50, start_token=2, end_token=3, pad_token=0
    ):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        memory = self.encode(src, src_mask)

        current_token = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=device
        )
        generated_tokens = []
        past_key_values = None
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, past_key_values = self.decode_with_cache(
                current_token, memory, past_key_values
            )

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            is_finished |= next_token.squeeze(-1) == end_token

            effective_token = next_token.masked_fill(
                is_finished.unsqueeze(-1), pad_token
            )
            generated_tokens.append(effective_token)
            current_token = effective_token

            if is_finished.all():
                break

        return torch.cat(generated_tokens, dim=1)

    def decode_with_cache(self, tgt, memory, past_key_values=None):
        x = self.dropout(self.tgt_embedding(tgt))

        new_past_key_values = []
        for i, layer in enumerate(self.decoder_layers):
            layer_past = past_key_values[i] if past_key_values is not None else None

            x, present_cache = layer(
                x,
                memory,
                causal_mask=None,  # single_step decoding, no causal mask needed
                past_key_value=layer_past,
                use_cache=True,
            )
            new_past_key_values.append(present_cache)

        return self.final_linear(x), new_past_key_values

    @staticmethod
    def _make_causal_mask(tgt, tgt_mask=None):
        device = tgt.device
        seq_len = tgt.size(1)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        causal_mask = (mask == 0).unsqueeze(0).unsqueeze(0)

        if tgt_mask is not None:
            padding_mask = tgt_mask.bool().unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & padding_mask

        return causal_mask
