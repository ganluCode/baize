"""Unit tests for baize.user.api_key."""

from baize.user.api_key import generate_api_key, hash_api_key, verify_api_key


def test_generate_api_key_returns_different_values_each_call():
    key1 = generate_api_key()
    key2 = generate_api_key()
    assert key1 != key2


def test_generate_api_key_length_at_least_32():
    key = generate_api_key()
    assert len(key) >= 32


def test_hash_api_key_differs_from_plaintext():
    key = generate_api_key()
    assert hash_api_key(key) != key


def test_verify_api_key_correct_key_returns_true():
    key = generate_api_key()
    stored = hash_api_key(key)
    assert verify_api_key(key, stored) is True


def test_verify_api_key_wrong_key_returns_false():
    key = generate_api_key()
    stored = hash_api_key(key)
    assert verify_api_key("wrong-key", stored) is False


def test_hash_api_key_same_input_different_results_due_to_salt():
    key = generate_api_key()
    hash1 = hash_api_key(key)
    hash2 = hash_api_key(key)
    assert hash1 != hash2


def test_verify_api_key_with_two_different_hashes_of_same_key():
    key = generate_api_key()
    hash1 = hash_api_key(key)
    hash2 = hash_api_key(key)
    assert verify_api_key(key, hash1) is True
    assert verify_api_key(key, hash2) is True
