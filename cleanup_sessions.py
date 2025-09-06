"""
Manual session cleanup script.
Run this script to clean up old sessions and results.
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_sessions(days_old=7, dry_run=True):
    """
    Clean up old session files and results.
    
    Args:
        days_old: Number of days old to consider for cleanup
        dry_run: If True, only show what would be deleted without actually deleting
    """
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    # Directories to clean
    directories = ['sessions', 'results']
    
    total_deleted = 0
    total_size = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist, skipping...")
            continue
            
        print(f"\n📁 Cleaning {directory}/ directory...")
        
        for file_path in Path(directory).glob("*"):
            try:
                # Get file modification time
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                file_size = file_path.stat().st_size
                
                if file_time < cutoff_date:
                    if dry_run:
                        print(f"  Would delete: {file_path} (modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}, size: {file_size} bytes)")
                    else:
                        file_path.unlink()
                        print(f"  ✅ Deleted: {file_path}")
                    
                    total_deleted += 1
                    total_size += file_size
                else:
                    print(f"  Keeping: {file_path} (modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
                    
            except Exception as e:
                print(f"  ❌ Error processing {file_path}: {str(e)}")
    
    print(f"\n📊 Summary:")
    print(f"  Files {'would be' if dry_run else ''} deleted: {total_deleted}")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
    
    if dry_run:
        print(f"\n💡 To actually delete these files, run:")
        print(f"   python cleanup_sessions.py --execute")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old session files")
    parser.add_argument("--days", type=int, default=7, help="Number of days old to consider for cleanup (default: 7)")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry run)")
    
    args = parser.parse_args()
    
    print("🧹 Session Cleanup Tool")
    print("=" * 50)
    print(f"Cleaning files older than {args.days} days")
    print(f"Mode: {'DRY RUN' if not args.execute else 'EXECUTE'}")
    print("=" * 50)
    
    cleanup_sessions(days_old=args.days, dry_run=not args.execute)
    
    if not args.execute:
        print(f"\n⚠️  This was a dry run. No files were actually deleted.")
        print(f"   Use --execute flag to actually delete the files.")

if __name__ == "__main__":
    main()
