import pytest
import torch

from mae_ast.models.positional import build_1d_sincos_pos_embed


def test_positional_embedding_shape():
    positional_embedding = build_1d_sincos_pos_embed(
        embed_dim=768,
        seq_len=512,
    )

    assert positional_embedding.shape == (1, 512, 768)


def test_positional_embedding_is_not_trainable():
    positional_embedding = build_1d_sincos_pos_embed(
        embed_dim=768,
        seq_len=512,
    )

    assert positional_embedding.requires_grad is False


def test_first_position():
    positional_embedding = build_1d_sincos_pos_embed(
        embed_dim=8,
        seq_len=4,
    )

    first_position = positional_embedding[0, 0]

    # Alla posizione zero:
    #
    # sin(0) = 0
    # cos(0) = 1
    expected = torch.tensor(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    )

    assert torch.allclose(first_position, expected)


def test_different_positions_have_different_embeddings():
    positional_embedding = build_1d_sincos_pos_embed(
        embed_dim=768,
        seq_len=512,
    )

    position_0 = positional_embedding[0, 0]
    position_1 = positional_embedding[0, 1]

    assert not torch.equal(position_0, position_1)


def test_odd_embedding_dimension_raises_error():
    with pytest.raises(ValueError):
        build_1d_sincos_pos_embed(
            embed_dim=767,
            seq_len=512,
        )


def test_invalid_sequence_length_raises_error():
    with pytest.raises(ValueError):
        build_1d_sincos_pos_embed(
            embed_dim=768,
            seq_len=0,
        )