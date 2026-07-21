"""Regression checks for README content that must be visible on Read the Docs."""

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).parents[1]
README_IMAGE = re.compile(r"!\[[^]]*\]\((?!https?://)([^)]+)\)")


def test_every_readme_image_is_referenced_by_a_sphinx_source():
    """Keep local README figures discoverable in the published documentation."""
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    local_images = {
        Path(match).as_posix()
        for match in README_IMAGE.findall(readme)
    }

    docs_text = "\n".join(
        path.read_text()
        for path in sorted((REPOSITORY_ROOT / "docs").rglob("*"))
        if path.suffix in {".md", ".rst"}
    )

    missing = sorted(
        image
        for image in local_images
        if f"../{image}" not in docs_text
    )
    assert not missing, (
        "README images missing from Read the Docs sources: "
        + ", ".join(missing)
    )
