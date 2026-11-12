"""Model Artifact Integrity Validation and Serialization Security.

Demonstrates:
1. Cryptographic SHA-256 verification prior to deserializing any pickle artifact.
2. Blocking tampered or untrusted model files before pickle.load() executes.
3. Explaining arbitrary code execution risks in Python pickle/joblib deserialization.
"""

import os
import pickle
import tempfile
from typing import Any

from reproducible_training import compute_file_sha256, train_and_save_pipeline


class ArtifactSecurityError(SecurityError if hasattr(__builtins__, "SecurityError") else Exception):
    """Raised when a model artifact fails SHA-256 checksum verification."""
    pass


def safe_load_trusted_artifact(artifact_path: str, expected_sha256: str) -> Any:
    """Verifies cryptographic checksum BEFORE deserializing pickle binary."""
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    actual_sha256 = compute_file_sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ArtifactSecurityError(
            f"INTEGRITY VIOLATION: Checksum mismatch for '{artifact_path}'. "
            f"Expected {expected_sha256[:16]}..., got {actual_sha256[:16]}... "
            "Refusing to execute pickle.load() to prevent arbitrary code execution!"
        )

    with open(artifact_path, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    print("=" * 75)
    print("CHAPTER 36: MODEL ARTIFACT INTEGRITY & SERIALIZATION SECURITY")
    print("=" * 75)

    res = train_and_save_pipeline("logistic_regression", C=1.0, seed=42, artifact_filename="trusted_v1.pkl")
    artifact_path = res["artifact_path"]
    valid_hash = res["checksum_sha256"]

    # 1. Valid Trusted Load
    print("1. Verifying trusted local artifact...")
    loaded_model = safe_load_trusted_artifact(artifact_path, valid_hash)
    print(f"   [PASS] Loaded model safely: {type(loaded_model).__name__} (Checksum verified)")

    # 2. Tampered Artifact Simulation
    print("\n2. Simulating tampered / corrupted artifact detection...")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
        with open(artifact_path, "rb") as orig:
            tmp.write(orig.read() + b"TAMPERED_MALICIOUS_BYTES")

    try:
        safe_load_trusted_artifact(tmp_path, valid_hash)
    except ArtifactSecurityError as e:
        print(f"   [BLOCKED] {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\nSecurity Principle: Python's pickle/joblib executes arbitrary opcodes on load.")
    print("Never call pickle.load() on unverified or untrusted user-uploaded artifacts.")
    print("[SUCCESS] Artifact integrity validation verified.")


if __name__ == "__main__":
    main()

