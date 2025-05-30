import os
import shutil

def restore_from_backup(corrupted_folder_path, backup_folder_path):
    # Check if corrupted folder exists
    if os.path.exists(corrupted_folder_path):
        print(f"Deleting corrupted folder: {corrupted_folder_path}")
        shutil.rmtree(corrupted_folder_path)
    else:
        print(f"Corrupted folder not found. Will restore anyway: {corrupted_folder_path}")

    # Copy backup to original location
    print(f"Restoring from backup: {backup_folder_path} -> {corrupted_folder_path}")
    shutil.copytree(backup_folder_path, corrupted_folder_path)

    print("Restoration complete.")

corrupted_folder = "knowledgebase2"
backup_folder = "db_backup/knowledgebase2"

restore_from_backup(corrupted_folder, backup_folder)
