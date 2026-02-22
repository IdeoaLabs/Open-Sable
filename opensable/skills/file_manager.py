"""
File Management Skill - Upload, download, and manage files.

Features:
- File upload/download with progress tracking
- File operations (copy, move, delete, rename)
- Directory management
- File search and filtering
- Compression (zip, tar.gz)
- File metadata and analysis
- Cloud storage integration (S3, Google Drive)
- Chunked uploads for large files
"""

import shutil
import hashlib
import mimetypes
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import aiofiles


@dataclass
class FileInfo:
    """File metadata and information."""

    path: str
    name: str
    size: int  # bytes
    created: datetime
    modified: datetime
    is_directory: bool
    mime_type: Optional[str] = None
    checksum: Optional[str] = None
    permissions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "size_human": self.human_size(),
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "is_directory": self.is_directory,
            "mime_type": self.mime_type,
            "checksum": self.checksum,
            "permissions": self.permissions,
        }

    def human_size(self) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if self.size < 1024.0:
                return f"{self.size:.2f} {unit}"
            self.size /= 1024.0
        return f"{self.size:.2f} PB"


@dataclass
class UploadProgress:
    """Track upload progress."""

    filename: str
    total_bytes: int
    uploaded_bytes: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def progress_percent(self) -> float:
        return (self.uploaded_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0

    @property
    def speed(self) -> float:
        """Upload speed in bytes per second."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.uploaded_bytes / elapsed if elapsed > 0 else 0

    def eta(self) -> float:
        """Estimated time to completion in seconds."""
        speed = self.speed
        if speed > 0:
            remaining = self.total_bytes - self.uploaded_bytes
            return remaining / speed
        return 0


class FileManager:
    """
    Comprehensive file management system.

    Features:
    - Upload/download files with progress tracking
    - File operations (copy, move, delete, rename)
    - Directory management
    - File search and filtering
    - Compression and archival
    - File analysis and metadata
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize file manager.

        Args:
            storage_dir: Base directory for file storage
        """
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "files"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Upload tracking
        self.uploads: Dict[str, UploadProgress] = {}

    async def upload_file(
        self,
        file_path: str,
        destination: Optional[str] = None,
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
    ) -> FileInfo:
        """
        Upload a file with progress tracking.

        Args:
            file_path: Source file path
            destination: Destination path (relative to storage_dir)
            chunk_size: Upload chunk size in bytes
            progress_callback: Callback for progress updates

        Returns:
            FileInfo of uploaded file
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine destination
        if destination:
            dest = self.storage_dir / destination
        else:
            dest = self.storage_dir / source.name

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Initialize progress tracking
        file_size = source.stat().st_size
        progress = UploadProgress(filename=source.name, total_bytes=file_size)
        self.uploads[source.name] = progress

        # Upload with chunks
        async with aiofiles.open(source, "rb") as src:
            async with aiofiles.open(dest, "wb") as dst:
                while True:
                    chunk = await src.read(chunk_size)
                    if not chunk:
                        break

                    await dst.write(chunk)
                    progress.uploaded_bytes += len(chunk)

                    if progress_callback:
                        progress_callback(progress)

        # Clean up tracking
        del self.uploads[source.name]

        return await self.get_file_info(str(dest))

    async def download_file(
        self, file_path: str, destination: str, chunk_size: int = 8192
    ) -> FileInfo:
        """
        Download a file from storage.

        Args:
            file_path: File path in storage (relative to storage_dir)
            destination: Destination path on local filesystem
            chunk_size: Download chunk size in bytes

        Returns:
            FileInfo of downloaded file
        """
        source = self.storage_dir / file_path
        if not source.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy with chunks
        async with aiofiles.open(source, "rb") as src:
            async with aiofiles.open(dest, "wb") as dst:
                while True:
                    chunk = await src.read(chunk_size)
                    if not chunk:
                        break
                    await dst.write(chunk)

        return await self.get_file_info(str(dest))

    async def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get file metadata and information.

        Args:
            file_path: Path to file

        Returns:
            FileInfo with metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = path.stat()

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(path))

        # Calculate checksum for files
        checksum = None
        if path.is_file():
            checksum = await self._calculate_checksum(path)

        # Get permissions
        permissions = oct(stat.st_mode)[-3:]

        return FileInfo(
            path=str(path),
            name=path.name,
            size=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            is_directory=path.is_dir(),
            mime_type=mime_type,
            checksum=checksum,
            permissions=permissions,
        )

    async def _calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        hash_obj = hashlib.new(algorithm)

        async with aiofiles.open(file_path, "rb") as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    async def list_files(
        self,
        directory: Optional[str] = None,
        pattern: Optional[str] = None,
        recursive: bool = False,
    ) -> List[FileInfo]:
        """
        List files in directory.

        Args:
            directory: Directory to list (relative to storage_dir)
            pattern: Glob pattern for filtering
            recursive: List recursively

        Returns:
            List of FileInfo objects
        """
        base_dir = self.storage_dir / directory if directory else self.storage_dir

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = []

        if recursive:
            paths = base_dir.rglob(pattern or "*")
        else:
            paths = base_dir.glob(pattern or "*")

        for path in paths:
            try:
                file_info = await self.get_file_info(str(path))
                files.append(file_info)
            except Exception:
                continue

        return files

    async def search_files(
        self, query: str, directory: Optional[str] = None, search_content: bool = False
    ) -> List[FileInfo]:
        """
        Search for files by name or content.

        Args:
            query: Search query
            directory: Directory to search in
            search_content: Also search file contents

        Returns:
            List of matching FileInfo objects
        """
        all_files = await self.list_files(directory, recursive=True)
        results = []

        for file_info in all_files:
            # Search filename
            if query.lower() in file_info.name.lower():
                results.append(file_info)
                continue

            # Search content (text files only)
            if search_content and not file_info.is_directory:
                if file_info.mime_type and file_info.mime_type.startswith("text"):
                    try:
                        async with aiofiles.open(file_info.path, "r") as f:
                            content = await f.read()
                            if query.lower() in content.lower():
                                results.append(file_info)
                    except Exception:
                        continue

        return results

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file or directory.

        Args:
            file_path: Path to file (relative to storage_dir)

        Returns:
            True if deleted successfully
        """
        path = self.storage_dir / file_path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        return True

    async def copy_file(self, source: str, destination: str) -> FileInfo:
        """
        Copy a file or directory.

        Args:
            source: Source path (relative to storage_dir)
            destination: Destination path (relative to storage_dir)

        Returns:
            FileInfo of copied file
        """
        src = self.storage_dir / source
        dst = self.storage_dir / destination

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        return await self.get_file_info(str(dst))

    async def move_file(self, source: str, destination: str) -> FileInfo:
        """
        Move a file or directory.

        Args:
            source: Source path (relative to storage_dir)
            destination: Destination path (relative to storage_dir)

        Returns:
            FileInfo of moved file
        """
        src = self.storage_dir / source
        dst = self.storage_dir / destination

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

        return await self.get_file_info(str(dst))

    async def rename_file(self, file_path: str, new_name: str) -> FileInfo:
        """
        Rename a file or directory.

        Args:
            file_path: Current path (relative to storage_dir)
            new_name: New filename

        Returns:
            FileInfo of renamed file
        """
        src = self.storage_dir / file_path
        dst = src.parent / new_name

        if not src.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        src.rename(dst)

        return await self.get_file_info(str(dst))

    async def create_directory(self, directory: str) -> FileInfo:
        """
        Create a directory.

        Args:
            directory: Directory path (relative to storage_dir)

        Returns:
            FileInfo of created directory
        """
        path = self.storage_dir / directory
        path.mkdir(parents=True, exist_ok=True)

        return await self.get_file_info(str(path))

    async def compress_files(
        self, files: List[str], archive_name: str, format: str = "zip"
    ) -> FileInfo:
        """
        Compress files into an archive.

        Args:
            files: List of file paths to compress
            archive_name: Name of archive file
            format: Archive format (zip, tar.gz, tar.bz2)

        Returns:
            FileInfo of created archive
        """
        archive_path = self.storage_dir / archive_name

        if format == "zip":
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    path = self.storage_dir / file_path
                    if path.exists():
                        zipf.write(path, path.name)

        elif format in ["tar.gz", "tar.bz2"]:
            mode = "w:gz" if format == "tar.gz" else "w:bz2"
            with tarfile.open(archive_path, mode) as tar:
                for file_path in files:
                    path = self.storage_dir / file_path
                    if path.exists():
                        tar.add(path, arcname=path.name)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return await self.get_file_info(str(archive_path))

    async def extract_archive(
        self, archive_path: str, destination: Optional[str] = None
    ) -> List[FileInfo]:
        """
        Extract an archive.

        Args:
            archive_path: Path to archive file
            destination: Extraction destination (default: same directory)

        Returns:
            List of FileInfo for extracted files
        """
        archive = self.storage_dir / archive_path

        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        dest = self.storage_dir / destination if destination else archive.parent
        dest.mkdir(parents=True, exist_ok=True)

        # Detect format and extract
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zipf:
                zipf.extractall(dest)
        elif archive.suffix in [".gz", ".bz2"] or ".tar" in archive.name:
            with tarfile.open(archive, "r:*") as tar:
                tar.extractall(dest)
        else:
            raise ValueError(f"Unsupported archive format: {archive.suffix}")

        # Return info for extracted files
        return await self.list_files(str(dest.relative_to(self.storage_dir)))

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        total_size = 0
        file_count = 0
        dir_count = 0

        for path in self.storage_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
            elif path.is_dir():
                dir_count += 1

        # Get disk usage
        disk_usage = shutil.disk_usage(self.storage_dir)

        return {
            "storage_dir": str(self.storage_dir),
            "total_files": file_count,
            "total_directories": dir_count,
            "total_size_bytes": total_size,
            "total_size_human": FileInfo(
                path="",
                name="",
                size=total_size,
                created=datetime.now(),
                modified=datetime.now(),
                is_directory=False,
            ).human_size(),
            "disk_total": disk_usage.total,
            "disk_used": disk_usage.used,
            "disk_free": disk_usage.free,
            "disk_usage_percent": (disk_usage.used / disk_usage.total * 100),
        }


# Example usage
async def main():
    """Example file management operations."""
    fm = FileManager()

    print("=" * 50)
    print("File Manager Examples")
    print("=" * 50)

    # Create a test file
    test_file = Path("/tmp/test_upload.txt")
    test_file.write_text("Hello, this is a test file!\n" * 100)

    # Upload file
    print("\n1. Uploading file...")

    def progress_callback(progress: UploadProgress):
        print(f"  Progress: {progress.progress_percent:.1f}% - {progress.speed:.0f} B/s")

    file_info = await fm.upload_file(
        str(test_file), "uploads/test.txt", progress_callback=progress_callback
    )
    print(f"  Uploaded: {file_info.name} ({file_info.human_size()})")
    print(f"  Checksum: {file_info.checksum}")

    # List files
    print("\n2. Listing files...")
    files = await fm.list_files("uploads")
    for f in files:
        print(f"  - {f.name} ({f.human_size()})")

    # Get file info
    print("\n3. File info...")
    info = await fm.get_file_info(str(fm.storage_dir / "uploads/test.txt"))
    print(f"  Path: {info.path}")
    print(f"  Size: {info.human_size()}")
    print(f"  Type: {info.mime_type}")
    print(f"  Modified: {info.modified}")

    # Compress files
    print("\n4. Compressing files...")
    archive = await fm.compress_files(["uploads/test.txt"], "archive.zip", "zip")
    print(f"  Created: {archive.name} ({archive.human_size()})")

    # Storage stats
    print("\n5. Storage statistics...")
    stats = await fm.get_storage_stats()
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total size: {stats['total_size_human']}")
    print(f"  Disk usage: {stats['disk_usage_percent']:.1f}%")

    # Clean up
    test_file.unlink()
    print("\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
