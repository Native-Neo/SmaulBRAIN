from smaulbrain.data import ByteStream


def test_resume_accepts_symlink_to_same_corpus(tmp_path):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(bytes(range(64)))
    alias = tmp_path / "alias.bin"
    alias.symlink_to(corpus)

    original = ByteStream(path=str(corpus))
    original.next_tokens(8)
    state = original.state_dict()

    resumed = ByteStream(path=str(alias))
    resumed.load_state_dict(state)

    assert resumed.cursor == 9
