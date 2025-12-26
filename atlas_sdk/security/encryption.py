"""
Data Encryption Module

Provides encryption at rest for sensitive experiment data.
Uses industry-standard AES-256-GCM encryption.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Cryptography is optional
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


@dataclass
class EncryptionKey:
    """Encryption key with metadata."""
    key_id: str
    key: bytes
    created_at: str
    algorithm: str = "AES-256-GCM"


class DataEncryption:
    """
    Enterprise data encryption for Atlas.

    Uses AES-256-GCM for authenticated encryption.
    Supports key derivation from passwords and key rotation.
    """

    NONCE_SIZE = 12
    TAG_SIZE = 16
    KEY_SIZE = 32  # 256 bits

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption with a key.

        Args:
            key: 32-byte encryption key. If None, generates a new one.
        """
        if not HAS_CRYPTO:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )

        self.key = key or self._generate_key()
        self._cipher = AESGCM(self.key)

    @classmethod
    def from_password(
        cls,
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000,
    ) -> Tuple["DataEncryption", bytes]:
        """
        Create encryption from a password using PBKDF2.

        Args:
            password: User password
            salt: Salt for key derivation (generated if None)
            iterations: PBKDF2 iterations

        Returns:
            Tuple of (DataEncryption instance, salt)
        """
        if not HAS_CRYPTO:
            raise ImportError("cryptography package required")

        salt = salt or os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=cls.KEY_SIZE,
            salt=salt,
            iterations=iterations,
        )

        key = kdf.derive(password.encode())
        return cls(key), salt

    def _generate_key(self) -> bytes:
        """Generate a cryptographically secure key."""
        return secrets.token_bytes(self.KEY_SIZE)

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt data with AES-256-GCM.

        Args:
            plaintext: Data to encrypt

        Returns:
            nonce + ciphertext + tag (concatenated)
        """
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        ciphertext = self._cipher.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.

        Args:
            ciphertext: nonce + encrypted data + tag

        Returns:
            Decrypted plaintext
        """
        nonce = ciphertext[:self.NONCE_SIZE]
        actual_ciphertext = ciphertext[self.NONCE_SIZE:]
        return self._cipher.decrypt(nonce, actual_ciphertext, None)

    def encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Encrypt a file."""
        with open(input_path, "rb") as f:
            plaintext = f.read()

        ciphertext = self.encrypt(plaintext)

        with open(output_path, "wb") as f:
            f.write(ciphertext)

    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Decrypt a file."""
        with open(input_path, "rb") as f:
            ciphertext = f.read()

        plaintext = self.decrypt(ciphertext)

        with open(output_path, "wb") as f:
            f.write(plaintext)

    def get_key_fingerprint(self) -> str:
        """Get a fingerprint of the current key for identification."""
        return hashlib.sha256(self.key).hexdigest()[:16]

    @staticmethod
    def generate_key_file(path: Path) -> bytes:
        """Generate and save a new encryption key."""
        key = secrets.token_bytes(DataEncryption.KEY_SIZE)

        # Save key with restrictive permissions
        with open(path, "wb") as f:
            f.write(key)

        os.chmod(path, 0o600)
        return key

    @staticmethod
    def load_key_file(path: Path) -> bytes:
        """Load encryption key from file."""
        with open(path, "rb") as f:
            return f.read()


class SecureStorage:
    """
    Encrypted storage wrapper for sensitive data.

    Provides transparent encryption/decryption for stored data.
    """

    def __init__(self, storage_path: Path, encryption: DataEncryption):
        self.storage_path = storage_path
        self.encryption = encryption
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        """Store encrypted data."""
        encrypted = self.encryption.encrypt(data)
        file_path = self.storage_path / self._safe_filename(key)

        with open(file_path, "wb") as f:
            f.write(encrypted)

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve and decrypt data."""
        file_path = self.storage_path / self._safe_filename(key)

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            encrypted = f.read()

        return self.encryption.decrypt(encrypted)

    def delete(self, key: str) -> bool:
        """Delete stored data."""
        file_path = self.storage_path / self._safe_filename(key)

        if file_path.exists():
            # Secure delete: overwrite with random data
            size = file_path.stat().st_size
            with open(file_path, "wb") as f:
                f.write(secrets.token_bytes(size))
            file_path.unlink()
            return True

        return False

    def _safe_filename(self, key: str) -> str:
        """Convert key to safe filename."""
        return hashlib.sha256(key.encode()).hexdigest()

    def list_keys(self) -> list:
        """List all stored keys (hashed filenames)."""
        return [f.name for f in self.storage_path.iterdir() if f.is_file()]
