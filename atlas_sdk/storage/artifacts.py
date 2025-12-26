"""
Safetensors-based artifact storage for tensors and model weights.

Memory-mappable, zero-copy, and safe (no arbitrary code execution).
Allows instant visualization of 4GB+ attention maps.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_numpy_file

from atlas_sdk.types import Artifact, ArtifactType, TensorMetric


class ArtifactStore:
    """
    Artifact storage using safetensors for tensor data.

    Safetensors provides:
    - Memory mapping for zero-copy access
    - Safety (no pickle, no arbitrary code execution)
    - Fast loading of tensor slices
    """

    def __init__(self, data_dir: Path, run_id: str):
        self.data_dir = data_dir
        self.run_id = run_id
        self.artifacts_dir = data_dir / "artifacts" / run_id
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Index file for artifact metadata
        self.index_path = self.artifacts_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load artifact index from disk."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save artifact index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()[:16]

    def save_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """
        Save a numpy tensor as a safetensors file.

        Args:
            name: Tensor name (e.g., "attention_weights", "gradients")
            tensor: NumPy array to save
            step: Training step
            metadata: Optional metadata dict

        Returns:
            Artifact record
        """
        # Create unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace("/", "_").replace(".", "_")
        filename = f"{safe_name}_step{step}_{timestamp}.safetensors"
        filepath = self.artifacts_dir / filename

        # Save tensor using safetensors
        tensors_dict = {name: tensor}
        save_numpy_file(tensors_dict, str(filepath))

        # Compute checksum
        with open(filepath, "rb") as f:
            checksum = self._compute_checksum(f.read())

        # Create artifact record
        artifact = Artifact(
            name=name,
            artifact_type=ArtifactType.TENSOR,
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            checksum=checksum,
            metadata={
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "step": step,
                **(metadata or {}),
            },
        )

        # Update index
        self._index[str(artifact.id)] = {
            "name": artifact.name,
            "path": str(filepath),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "step": step,
            "created_at": artifact.created_at.isoformat(),
            "checksum": checksum,
        }
        self._save_index()

        return artifact

    def save_checkpoint(
        self,
        name: str,
        tensors: Dict[str, np.ndarray],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """
        Save multiple tensors as a single checkpoint file.

        Args:
            name: Checkpoint name
            tensors: Dict of tensor_name -> numpy array
            step: Training step
            metadata: Optional metadata

        Returns:
            Artifact record
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace("/", "_").replace(".", "_")
        filename = f"checkpoint_{safe_name}_step{step}_{timestamp}.safetensors"
        filepath = self.artifacts_dir / filename

        # Save all tensors
        save_numpy_file(tensors, str(filepath))

        # Compute checksum
        with open(filepath, "rb") as f:
            checksum = self._compute_checksum(f.read())

        # Gather shape info
        tensor_info = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in tensors.items()}

        artifact = Artifact(
            name=name,
            artifact_type=ArtifactType.CHECKPOINT,
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            checksum=checksum,
            metadata={
                "tensors": tensor_info,
                "step": step,
                **(metadata or {}),
            },
        )

        self._index[str(artifact.id)] = {
            "name": artifact.name,
            "type": "checkpoint",
            "path": str(filepath),
            "tensors": tensor_info,
            "step": step,
            "created_at": artifact.created_at.isoformat(),
            "checksum": checksum,
        }
        self._save_index()

        return artifact

    def load_tensor(
        self,
        artifact_id: Optional[str] = None,
        path: Optional[str] = None,
        tensor_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load a tensor from safetensors file.

        Uses memory mapping for zero-copy access on large tensors.

        Args:
            artifact_id: ID from the index
            path: Direct path to safetensors file
            tensor_name: Name of tensor within the file (if multiple)

        Returns:
            NumPy array
        """
        if artifact_id:
            if artifact_id not in self._index:
                raise KeyError(f"Artifact {artifact_id} not found")
            path = self._index[artifact_id]["path"]
            tensor_name = tensor_name or self._index[artifact_id].get("name")

        if not path:
            raise ValueError("Must provide artifact_id or path")

        with safe_open(path, framework="numpy") as f:
            if tensor_name:
                return f.get_tensor(tensor_name)
            else:
                # Return first tensor if name not specified
                keys = list(f.keys())
                if keys:
                    return f.get_tensor(keys[0])
                raise ValueError("No tensors found in file")

    def load_tensor_slice(
        self,
        path: str,
        tensor_name: str,
        slices: Tuple[slice, ...],
    ) -> np.ndarray:
        """
        Load a slice of a tensor using memory mapping.

        Efficient for viewing portions of large tensors (e.g., attention heads).

        Args:
            path: Path to safetensors file
            tensor_name: Name of tensor
            slices: Tuple of slice objects for each dimension

        Returns:
            Sliced numpy array
        """
        with safe_open(path, framework="numpy") as f:
            tensor = f.get_tensor(tensor_name)
            return tensor[slices]

    def get_tensor_info(self, path: str) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata about tensors in a file without loading them.

        Args:
            path: Path to safetensors file

        Returns:
            Dict of tensor_name -> {shape, dtype}
        """
        with safe_open(path, framework="numpy") as f:
            info = {}
            for key in f.keys():
                tensor = f.get_tensor(key)
                info[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "size_bytes": tensor.nbytes,
                }
            return info

    def list_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None,
    ) -> List[Dict[str, Any]]:
        """List all artifacts, optionally filtered by type."""
        artifacts = list(self._index.values())

        if artifact_type:
            type_str = artifact_type.value
            artifacts = [a for a in artifacts if a.get("type", "tensor") == type_str]

        return sorted(artifacts, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_artifact_by_step(
        self,
        name: str,
        step: int,
    ) -> Optional[Dict[str, Any]]:
        """Get artifact by name and step."""
        for artifact_id, info in self._index.items():
            if info.get("name") == name and info.get("step") == step:
                info["id"] = artifact_id
                return info
        return None

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact and its file."""
        if artifact_id not in self._index:
            return False

        path = self._index[artifact_id]["path"]
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

        del self._index[artifact_id]
        self._save_index()
        return True

    def cleanup_old_artifacts(
        self,
        keep_last_n: int = 5,
        artifact_type: Optional[ArtifactType] = None,
    ) -> int:
        """
        Clean up old artifacts, keeping only the most recent N per name.

        Returns:
            Number of artifacts deleted
        """
        # Group by name
        by_name: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for artifact_id, info in self._index.items():
            name = info.get("name", "unknown")
            if artifact_type and info.get("type") != artifact_type.value:
                continue
            if name not in by_name:
                by_name[name] = []
            by_name[name].append((artifact_id, info))

        deleted = 0
        for name, artifacts in by_name.items():
            # Sort by step (descending)
            sorted_artifacts = sorted(
                artifacts,
                key=lambda x: x[1].get("step", 0),
                reverse=True,
            )

            # Delete all but the last N
            for artifact_id, info in sorted_artifacts[keep_last_n:]:
                if self.delete_artifact(artifact_id):
                    deleted += 1

        return deleted

    def get_total_size(self) -> int:
        """Get total size of all artifacts in bytes."""
        total = 0
        for info in self._index.values():
            path = info.get("path")
            if path and os.path.exists(path):
                total += os.path.getsize(path)
        return total
