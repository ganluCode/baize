"""API Key utilities: generation, hashing, and verification.

Keys are stored as ``salt$sha256(salt + key)`` so that the same plaintext
produces a different stored value on every call (salt is random).
"""

import hashlib
import secrets


def generate_api_key() -> str:
    """Generate a cryptographically random plaintext API key.

    Returns:
        URL-safe base64 string of at least 32 characters.
    """
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Hash an API key with a random salt using SHA-256.

    Args:
        key: Plaintext API key.

    Returns:
        String in the format ``<hex_salt>$<hex_digest>``.
    """
    salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + key).encode()).hexdigest()
    return f"{salt}${digest}"


def verify_api_key(key: str, stored_hash: str) -> bool:
    """Verify a plaintext API key against a stored hash.

    Args:
        key: Plaintext API key to check.
        stored_hash: Previously hashed value produced by :func:`hash_api_key`.

    Returns:
        True if the key matches, False otherwise.
    """
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    expected = hashlib.sha256((salt + key).encode()).hexdigest()
    return secrets.compare_digest(expected, digest)
