"""
Integration tests for Mini Transformer.

全コンポーネントを組み合わせた動作確認。
"""

import torch
import pytest

from src.tokenizer import CharTokenizer
from src.embedding import TokenEmbedding, PositionalEncoding
from src.attention import MultiHeadAttention, create_causal_mask
from src.feedforward import FeedForward
from src.transformer import TransformerBlock
from src.model import MiniGPT, ModelConfig
from src.dataset import TextDataset, create_dataloader
from src.generate import generate


class TestTokenizer:
    def test_build_vocab(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])
        assert tokenizer.vocab_size > 4  # 特殊トークン + 文字
    
    def test_encode_decode(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])
        
        text = "hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert decoded == text
    
    def test_unknown_char(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["abc"])
        
        encoded = tokenizer.encode("xyz")
        # 未知文字はUNKトークンになる
        assert all(id == tokenizer.unk_id for id in encoded)


class TestEmbedding:
    def test_token_embedding_shape(self):
        embed = TokenEmbedding(vocab_size=100, embed_dim=64)
        x = torch.randint(0, 100, (2, 10))  # (batch=2, seq=10)
        
        out = embed(x)
        assert out.shape == (2, 10, 64)
    
    def test_positional_encoding_shape(self):
        pe = PositionalEncoding(embed_dim=64, max_len=100)
        x = torch.randn(2, 10, 64)  # (batch=2, seq=10, dim=64)
        
        out = pe(x)
        assert out.shape == x.shape
    
    def test_positional_encoding_uniqueness(self):
        pe = PositionalEncoding(embed_dim=64, max_len=100)
        encoding = pe.get_encoding(10)
        
        # 各位置のエンコーディングは異なる
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(encoding[i], encoding[j])


class TestAttention:
    def test_multihead_attention_shape(self):
        mha = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        
        out, attn = mha(x)
        assert out.shape == (2, 10, 64)
        assert attn.shape == (2, 4, 10, 10)
    
    def test_attention_weights_sum_to_one(self):
        mha = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)  # dropoutを無効化
        x = torch.randn(2, 10, 64)
        
        _, attn = mha(x)
        
        # 各行の合計が1になる
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_causal_mask(self):
        mask = create_causal_mask(5)
        
        # 下三角行列
        expected = torch.tensor([
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])
        assert torch.equal(mask, expected)
    
    def test_causal_mask_prevents_future(self):
        mha = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(1, 5, 64)
        mask = create_causal_mask(5)
        
        _, attn = mha(x, mask)
        
        # 未来位置のAttention重みは0
        for i in range(5):
            for j in range(i + 1, 5):
                assert torch.allclose(attn[0, :, i, j], torch.zeros(4), atol=1e-5)


class TestFeedForward:
    def test_shape_preservation(self):
        ff = FeedForward(embed_dim=64, ff_dim=256)
        x = torch.randn(2, 10, 64)
        
        out = ff(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(2, 10, 64)
        
        out, attn = block(x)
        assert out.shape == x.shape
        assert attn.shape == (2, 4, 10, 10)


class TestMiniGPT:
    def test_forward_shape(self):
        config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_len=32
        )
        model = MiniGPT(config)
        
        x = torch.randint(0, 100, (2, 10))
        logits = model(x)
        
        assert logits.shape == (2, 10, 100)
    
    def test_forward_with_attention(self):
        config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_len=32
        )
        model = MiniGPT(config)
        
        x = torch.randint(0, 100, (2, 10))
        logits, attns = model(x, return_attention=True)
        
        assert logits.shape == (2, 10, 100)
        assert len(attns) == 2  # num_layers
    
    def test_parameter_count(self):
        config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_len=32
        )
        model = MiniGPT(config)
        
        params = model.count_parameters()
        assert params > 0


class TestDataset:
    def test_dataset_length(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world " * 100])
        
        dataset = TextDataset(["hello world " * 100], tokenizer, seq_len=10)
        assert len(dataset) > 0
    
    def test_dataset_item_shape(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world " * 100])
        
        dataset = TextDataset(["hello world " * 100], tokenizer, seq_len=10)
        x, y = dataset[0]
        
        assert x.shape == (10,)
        assert y.shape == (10,)


class TestGeneration:
    def test_generate(self):
        config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_len=32
        )
        model = MiniGPT(config)
        
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])
        
        # vocab_sizeを合わせる
        config.vocab_size = tokenizer.vocab_size
        model = MiniGPT(config)
        
        generated = generate(
            model, tokenizer, "hello",
            max_new_tokens=10,
            temperature=1.0
        )
        
        assert len(generated) >= len("hello")
    
    def test_greedy_deterministic(self):
        config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_len=32
        )
        
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])
        config.vocab_size = tokenizer.vocab_size
        model = MiniGPT(config)
        
        # Temperature=0で決定的
        gen1 = generate(model, tokenizer, "hello", max_new_tokens=5, temperature=0)
        gen2 = generate(model, tokenizer, "hello", max_new_tokens=5, temperature=0)
        
        assert gen1 == gen2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
