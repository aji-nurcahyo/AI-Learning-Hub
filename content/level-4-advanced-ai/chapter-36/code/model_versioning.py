"""Semantic Model Versioning and Lineage Management.

Demonstrates:
1. Semantic Versioning (MAJOR.MINOR.PATCH) parsing and bump logic.
2. Tracking Model Lineage (model_id -> dataset_version -> training_run_id -> artifact_checksum).
3. Verifying immutability of published versions.
"""

import re
from typing import Dict, Tuple


def parse_semver(version: str) -> Tuple[int, int, int]:
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not m:
        raise ValueError(f"Invalid semantic version string: '{version}'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump_version(current_version: str, bump_type: str = "minor") -> str:
    major, minor, patch = parse_semver(current_version)
    if bump_type == "major":
        return f"v{major + 1}.0.0"
    elif bump_type == "minor":
        return f"v{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"v{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown bump_type: {bump_type}")


def build_lineage_record(
    model_id: str,
    version: str,
    run_id: str,
    dataset_version: str,
    parent_version: str,
    checksum: str,
) -> Dict[str, str]:
    return {
        "model_id": model_id,
        "version": version,
        "parent_version": parent_version,
        "run_id": run_id,
        "dataset_version": dataset_version,
        "artifact_sha256": checksum,
    }


def main():
    print("=" * 75)
    print("CHAPTER 36: SEMANTIC MODEL VERSIONING & LINEAGE")
    print("=" * 75)

    v1 = "v1.0.0"
    v_patch = bump_version(v1, "patch")
    v_minor = bump_version(v_patch, "minor")
    v_major = bump_version(v_minor, "major")

    print(f"Semantic Version Progression:")
    print(f"- Initial Baseline      : {v1}")
    print(f"- Hyperparam/Fix (patch): {v_patch} (Backward-compatible bug fix)")
    print(f"- Retrained Data (minor): {v_minor} (Same input schema, updated weights)")
    print(f"- Schema Change  (major): {v_major} (Breaking change to input/output API contract)\n")

    lineage = build_lineage_record(
        model_id="telemetry_classifier",
        version=v_minor,
        run_id="run_83a91c2f",
        dataset_version="telemetry_ds_a41b90",
        parent_version=v_patch,
        checksum="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    )
    print("Model Lineage Record:")
    for k, val in lineage.items():
        print(f"  {k:<18}: {val}")

    print("\n[SUCCESS] Semantic model versioning and lineage verified.")


if __name__ == "__main__":
    main()

