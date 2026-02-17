"""
File Management Examples - Upload, download, and manage files.

Demonstrates file operations, compression, search, and storage management.
"""

import asyncio
from pathlib import Path
from skills.file_manager import FileManager


async def main():
    """Run file management examples."""
    
    print("=" * 60)
    print("File Management Examples")
    print("=" * 60)
    
    manager = FileManager()
    
    # Example 1: Upload file
    print("\n1. File Upload")
    print("-" * 40)
    
    test_content = b"Hello, Open-Sable! This is a test file."
    result = await manager.upload_file(
        filename="test.txt",
        content=test_content,
        metadata={"type": "example", "category": "test"}
    )
    
    print(f"Uploaded: {result.filename}")
    print(f"Size: {result.size} bytes")
    print(f"Path: {result.path}")
    
    # Example 2: Download file
    print("\n2. File Download")
    print("-" * 40)
    
    download = await manager.download_file("test.txt")
    print(f"Downloaded: {download.filename}")
    print(f"Content: {download.content.decode()[:50]}...")
    
    # Example 3: File operations
    print("\n3. File Operations")
    print("-" * 40)
    
    # Copy
    copy_result = await manager.copy_file("test.txt", "test_copy.txt")
    print(f"Copied to: {copy_result.path}")
    
    # Rename
    rename_result = await manager.rename_file("test_copy.txt", "test_renamed.txt")
    print(f"Renamed to: {rename_result.filename}")
    
    # Move
    await manager.create_directory("subdir")
    move_result = await manager.move_file("test_renamed.txt", "subdir/test_moved.txt")
    print(f"Moved to: {move_result.path}")
    
    # Example 4: Directory management
    print("\n4. Directory Management")
    print("-" * 40)
    
    await manager.create_directory("examples/nested/deep")
    files = await manager.list_directory("examples")
    print(f"Files in examples/: {len(files)}")
    for f in files[:5]:
        print(f"  - {f.name} ({f.size} bytes)")
    
    # Example 5: File search
    print("\n5. File Search")
    print("-" * 40)
    
    # Search by name
    search_results = await manager.search_files(pattern="*.txt")
    print(f"Found {len(search_results)} .txt files:")
    for f in search_results[:5]:
        print(f"  - {f.path}")
    
    # Example 6: Compression
    print("\n6. File Compression")
    print("-" * 40)
    
    # Create multiple test files
    for i in range(5):
        await manager.upload_file(
            filename=f"data_{i}.txt",
            content=f"Data file {i}" * 100
        )
    
    # Compress files
    archive = await manager.compress_files(
        files=["test.txt", "data_0.txt", "data_1.txt"],
        archive_name="backup.zip"
    )
    print(f"Created archive: {archive.path}")
    print(f"Archive size: {archive.size} bytes")
    
    # Extract archive
    extracted = await manager.extract_archive("backup.zip", "extracted/")
    print(f"Extracted {len(extracted)} files")
    
    # Example 7: File metadata
    print("\n7. File Metadata")
    print("-" * 40)
    
    metadata = await manager.get_file_metadata("test.txt")
    print(f"Filename: {metadata.name}")
    print(f"Size: {metadata.size} bytes")
    print(f"Type: {metadata.mime_type}")
    print(f"Checksum: {metadata.checksum}")
    print(f"Created: {metadata.created_at}")
    
    # Example 8: Storage statistics
    print("\n8. Storage Statistics")
    print("-" * 40)
    
    stats = await manager.get_storage_stats()
    print(f"Total files: {stats.total_files}")
    print(f"Total size: {stats.total_size / 1024:.2f} KB")
    print(f"Used space: {stats.used_space / 1024:.2f} KB")
    print(f"Available: {stats.available_space / (1024**3):.2f} GB")
    
    # Example 9: Batch upload
    print("\n9. Batch Upload with Progress")
    print("-" * 40)
    
    large_content = b"X" * (1024 * 100)  # 100KB
    
    async def progress_callback(progress):
        print(f"  Upload progress: {progress}%")
    
    result = await manager.upload_file(
        filename="large_file.bin",
        content=large_content,
        progress_callback=progress_callback
    )
    print(f"Uploaded large file: {result.size} bytes")
    
    # Cleanup
    print("\n10. Cleanup")
    print("-" * 40)
    
    await manager.delete_file("test.txt")
    await manager.delete_directory("subdir")
    print("Cleaned up test files")
    
    print("\n" + "=" * 60)
    print("✅ File management examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
