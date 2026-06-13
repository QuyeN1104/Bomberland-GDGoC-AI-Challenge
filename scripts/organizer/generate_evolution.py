import sqlite3
import random
import re
import os
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from competition.integrations.drive_upload import get_drive_service, _find_child_folder_id, _create_folder

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    
    # Manually parse .env to avoid requiring python-dotenv
    env_path = repo_root / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

    drive_folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not drive_folder_id:
        print("ERROR: DRIVE_FOLDER_ID not found in .env")
        return

    db_path = repo_root / "competition.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. build submission -> team name mapping
    cursor.execute("SELECT s.submission_id, t.team_name FROM submissions s JOIN teams t ON s.canonical_team_id = t.canonical_team_id")
    sub_to_team = {row[0]: row[1] for row in cursor.fetchall()}

    # 2. collect all GIF drive URLs and group by (team, date)
    cursor.execute("SELECT created_at, player_submission_ids_csv, gif_drive_url FROM match_results WHERE gif_drive_url IS NOT NULL")
    
    # groups[team_name][date_str] = [list of file_ids]
    groups = defaultdict(lambda: defaultdict(list))

    for created_at, players_csv, gif_url in cursor.fetchall():
        if not players_csv or not gif_url:
            continue
        # Parse date (YYYY-MM-DD)
        try:
            date_str = created_at.split('T')[0]
        except:
            date_str = created_at.split(' ')[0]

        # parse lấy drive ID. vd: https://drive.google.com/file/d/FILE_ID/view
        match = re.search(r'/d/([^/]+)/', gif_url)
        if not match:
            continue
        file_id = match.group(1)

        # assign file_id to all participating teams
        sub_ids = [sid.strip() for sid in players_csv.split(',')]
        for sid in sub_ids:
            team_name = sub_to_team.get(sid)
            if team_name:
                groups[team_name][date_str].append(file_id)

    # 3. connect drive API
    print("Connecting to Google Drive API...")
    try:
        service = get_drive_service()
    except Exception as e:
        print(f"Error authenticating: {e}")
        return

    # 4. find or create evolution folder
    print("Locating evolution root folder...")
    evolution_folder_id = _find_child_folder_id(service, drive_folder_id, "evolution")
    if not evolution_folder_id:
        print("Creating evolution folder...")
        evolution_folder_id = _create_folder(service, drive_folder_id, "evolution")
        
    print(f"Evolution folder ID: {evolution_folder_id}")

    # 5. loop through all teams
    for team_name, dates_dict in groups.items():
        # sanitize team name for folder creation
        safe_team_name = "".join(c for c in team_name if c.isalnum() or c in (' ', '_', '-')).strip()
        print(f"\nprocessing team: {safe_team_name}")
        
        # create team folder inside evolution/
        team_folder_id = _find_child_folder_id(service, evolution_folder_id, safe_team_name)
        if not team_folder_id:
            team_folder_id = _create_folder(service, evolution_folder_id, safe_team_name)
            
        # avoid duplicates, list existing files in the team folder
        existing_files = []
        try:
            response = service.files().list(
                q=f"'{team_folder_id}' in parents and trashed=false",
                fields="files(id, name)",
                pageSize=1000
            ).execute()
            existing_files = [f["name"] for f in response.get("files", [])]
        except Exception as e:
            print(f"Warning: Could not list files for {safe_team_name}: {e}")

        # loop through each date
        for date_str, file_ids in sorted(dates_dict.items()):
            # randomly select up to 2 gifs
            selected = random.sample(file_ids, min(2, len(file_ids)))
            
            for i, src_file_id in enumerate(selected):
                dest_name = f"{date_str}_match_{i+1}.gif"
                
                # Skip if already exists
                if dest_name in existing_files:
                    continue
                    
                print(f"  -> Copying {dest_name}...")
                try:
                    service.files().copy(
                        fileId=src_file_id,
                        body={
                            'name': dest_name,
                            'parents': [team_folder_id]
                        }
                    ).execute()
                except Exception as e:
                    print(f"     Failed to copy {dest_name}: {e}")

    print("\nEvolution complete!")

if __name__ == "__main__":
    main()
