#!/bin/bash
# Backup script for Open-Sable data

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/opensable_backup_$TIMESTAMP.tar.gz"

echo "📦 Creating backup..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup important data
tar -czf "$BACKUP_FILE" \
    data/ \
    config/ \
    .env \
    --exclude='data/vectordb/chroma.sqlite3-wal' \
    --exclude='data/vectordb/chroma.sqlite3-shm'

echo "✅ Backup created: $BACKUP_FILE"

# Keep only last 10 backups
cd "$BACKUP_DIR"
ls -t opensable_backup_*.tar.gz | tail -n +11 | xargs rm -f 2>/dev/null || true

echo "🗂️  Backup history:"
ls -lh opensable_backup_*.tar.gz 2>/dev/null || echo "No backups found"
