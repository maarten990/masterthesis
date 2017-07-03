from .. import data

def test_create_dictionary():
    vocab = data.create_dictionary('training_data', '1800*.xml')

    assert 0 not in vocab.idx_to_token
    assert 0 not in vocab.token_to_idx
    assert len(vocab.idx_to_token) == len(vocab.token_to_idx)

    for i in range(1, len(vocab.idx_to_token) + 1):
        word = vocab.idx_to_token[i]
        reverse_idx = vocab.token_to_idx[word]
        assert(i == reverse_idx)