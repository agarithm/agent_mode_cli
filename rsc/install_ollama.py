import os
import pathlib
import shutil
import tarfile

ARCHIVE_PATH = "/tmp/ollama.tgz"
EXTRACT_ROOT = pathlib.Path("/tmp/ollama-unpacked")


def main() -> None:
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        for member in archive.getmembers():
            target = EXTRACT_ROOT / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                os.symlink(member.linkname, target)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    main()
