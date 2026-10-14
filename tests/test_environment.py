"""Tests for the project environment."""

from pathlib import Path


def test_project_root_exists() -> None:
    """Verify that the project root contains its configuration."""
    project_root = Path(__file__).resolve().parents[1]

    assert project_root.exists()
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / ".gitignore").is_file()


def test_project_source_directory_exists() -> None:
    """Verify that the main source directory exists."""
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "src" / "knee_ai").is_dir()
