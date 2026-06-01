import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build

from competition.evaluation.ranking import RankingSystem
from competition.storage import SubmissionStore


DEFAULT_CREDENTIALS_FILE = "secrets/service_account_credentials.json"
DEFAULT_SHEET_RANGE = "Leaderboard!A1"


def _format_created_at(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def _ensure_sheet_tab(service, spreadsheet_id: str, sheet_name: str):
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id, fields="sheets.properties.title").execute()
    existing_titles = {sheet["properties"]["title"] for sheet in spreadsheet.get("sheets", [])}
    if sheet_name in existing_titles:
        return

    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_name,
                        }
                    }
                }
            ]
        },
    ).execute()


def _write_sheet_values(service, spreadsheet_id: str, sheet_name: str, values: list[list[str]]):
    # Use proper quoting for sheet names with spaces
    clear_range = f"'{sheet_name}'!A:Z"
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=clear_range,
    ).execute()
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def _get_sheet_id(service, spreadsheet_id: str, sheet_name: str) -> Optional[int]:
    """Return the numeric sheetId for a given tab name, or None if not found."""
    spreadsheet = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id, fields="sheets.properties"
    ).execute()
    for sheet in spreadsheet.get("sheets", []):
        if sheet["properties"]["title"] == sheet_name:
            return sheet["properties"]["sheetId"]
    return None


def _prepend_sheet_values(
    service, spreadsheet_id: str, sheet_name: str, values: list[list[str]]
):
    """Insert new rows at the top of a sheet (right below the header row).

    Uses insertDimension to push existing rows down, then writes data into
    the newly created blank rows.  This preserves the header at row 1 and
    keeps the newest entries at the top.
    """
    if not values:
        return

    sheet_id = _get_sheet_id(service, spreadsheet_id, sheet_name)
    if sheet_id is None:
        return

    num_rows = len(values)

    # Step 1: Insert blank rows right after the header (startIndex=1 is row 2)
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "requests": [
                {
                    "insertDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": 1,
                            "endIndex": 1 + num_rows,
                        },
                        "inheritFromBefore": False,
                    }
                }
            ]
        },
    ).execute()

    # Step 2: Write the new data into the freshly inserted rows
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!A2",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def _logs_sync_marker_path(db_path: str) -> Path:
    """Return the path of the sync marker file, stored next to competition.db."""
    return Path(db_path).resolve().parent / ".last_logs_sync"


def _read_logs_sync_marker(db_path: str) -> Optional[str]:
    """Read the last-synced timestamp from the marker file, or None if missing."""
    marker = _logs_sync_marker_path(db_path)
    if marker.exists():
        content = marker.read_text().strip()
        return content if content else None
    return None


def _write_logs_sync_marker(db_path: str, timestamp: str) -> None:
    """Persist the timestamp of the most recently synced match log."""
    marker = _logs_sync_marker_path(db_path)
    marker.write_text(timestamp + "\n")


def _leaderboard_headers() -> list[str]:
    return [
        "Rank",
        "Team",
        "Submission ID",
        "Mu",
        "Sigma",
        "Score",
        "Games",
        "Win Rate",
        "Wins",
        "Draws",
        "Losses",
        "Avg Rank",
        "Avg Steps",
        "Is Baseline",
        "Is Active",
        "Created At",
    ]


def build_leaderboard_values(db_path: str = "competition.db", include_baseline: bool = True) -> list[list[str]]:
    ranking = RankingSystem(db_path=db_path)
    rows = ranking.get_leaderboard(include_baseline=include_baseline)

    values = [_leaderboard_headers()]
    for idx, row in enumerate(rows, start=1):
        values.append(
            [
                str(idx),
                row["team_name"],
                row["submission_id"],
                f"{row['mu']:.4f}",
                f"{row['sigma']:.4f}",
                f"{row['score']:.4f}",
                str(row["n_games"]),
                f"{row['win_rate']:.4f}",
                str(row["wins"]),
                str(row["draws"]),
                str(row["losses"]),
                f"{row['avg_rank']:.4f}",
                f"{row['avg_steps']:.4f}",
                "1" if row["is_baseline"] else "0",
                "1" if row["is_active"] else "0",
                _format_created_at(row["created_at"]),
            ]
        )
    return values


def build_feedback_values(db_path: str = "competition.db") -> list[list[str]]:
    store = SubmissionStore(db_path=db_path)
    rows = store.list_feedback_submissions()

    values = [["Created At", "Submission ID", "Team", "Validation Status", "Reason"]]
    for row in rows:
        values.append(
            [
                _format_created_at(row["created_at"]),
                row["submission_id"],
                row["team_name"],
                row["validation_status"],
                row["validation_reason"] or "",
            ]
        )
    return values


def _logs_headers() -> list[str]:
    return ["Created At", "Match ID", "Submission IDs", "JSON Drive URL", "GIF Drive URL", "Match Type", "Seed"]


def _logs_row(row: dict) -> list[str]:
    return [
        _format_created_at(row["created_at"]),
        row["match_id"],
        row["player_submission_ids_csv"],
        row["json_drive_url"] or "",
        row["gif_drive_url"] or "",
        row["match_type"],
        "" if row["seed"] is None else str(row["seed"]),
    ]


def build_logs_values(db_path: str = "competition.db", since: Optional[str] = None) -> list[list[str]]:
    """Build log rows for the Google Sheet.

    Args:
        db_path: Path to the SQLite database.
        since: If provided, only include matches created after this timestamp.
               When set, the returned list contains data rows only (no header).
    """
    store = SubmissionStore(db_path=db_path)
    rows = store.list_match_results(since=since)

    if since is not None:
        # Incremental mode: data rows only, no header (will be prepended)
        return [_logs_row(row) for row in rows]
    else:
        # Full mode: header + all data rows
        values = [_logs_headers()]
        for row in rows:
            values.append(_logs_row(row))
        return values


def update_google_sheets(
    db_path: str = "competition.db",
    credentials_file: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    sheet_range: Optional[str] = None,
    include_baseline: bool = True,
    update_feedback: bool = True,
):
    _logger = logging.getLogger(__name__)

    credentials_file = credentials_file or os.getenv("LEADERBOARD_CREDENTIALS_FILE", DEFAULT_CREDENTIALS_FILE)
    spreadsheet_id = spreadsheet_id or os.getenv("LEADERBOARD_SPREADSHEET_ID", "")
    sheet_range = sheet_range or os.getenv("LEADERBOARD_SHEET_RANGE", DEFAULT_SHEET_RANGE)

    if not spreadsheet_id:
        return {
            "status": "skipped",
            "reason": "missing_spreadsheet_id",
        }

    if not os.path.exists(credentials_file):
        return {
            "status": "error",
            "reason": f"credentials_not_found:{credentials_file}",
        }

    values = build_leaderboard_values(db_path=db_path, include_baseline=include_baseline)
    feedback_values = build_feedback_values(db_path=db_path) if update_feedback else []

    # --- Incremental Logs sync ---
    last_sync = _read_logs_sync_marker(db_path)
    if last_sync:
        # Incremental: fetch only matches newer than the marker
        logs_values = build_logs_values(db_path=db_path, since=last_sync)
        logs_mode = "incremental"
    else:
        # First run (or after reset): fetch everything
        logs_values = build_logs_values(db_path=db_path)
        logs_mode = "full"

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        service = build("sheets", "v4", credentials=creds)

        sheet_name = sheet_range.split("!")[0] if "!" in sheet_range else sheet_range

        _write_sheet_values(service, spreadsheet_id, sheet_name, values)

        if update_feedback:
            _ensure_sheet_tab(service, spreadsheet_id, "Submissions Feedback")
            _write_sheet_values(service, spreadsheet_id, "Submissions Feedback", feedback_values)

        # --- Logs tab ---
        _ensure_sheet_tab(service, spreadsheet_id, "Logs")
        logs_written = 0
        if logs_mode == "full":
            # First run: clear + write header and all rows
            _write_sheet_values(service, spreadsheet_id, "Logs", logs_values)
            logs_written = len(logs_values) - 1  # subtract header
        elif logs_values:
            # Incremental: prepend only new rows at the top
            _prepend_sheet_values(service, spreadsheet_id, "Logs", logs_values)
            logs_written = len(logs_values)
        # else: no new matches, skip Logs API call entirely

        # Update the sync marker to the newest match timestamp
        if logs_values:
            # We need the raw (unformatted) created_at from the DB for the marker.
            # Query just the single newest match timestamp to avoid re-fetching all rows.
            store = SubmissionStore(db_path=db_path)
            newest = store.list_match_results(since=last_sync) if last_sync else store.list_match_results()
            if newest:
                # newest is ordered DESC, so index 0 has the latest created_at
                _write_logs_sync_marker(db_path, newest[0]["created_at"])
                _logger.info("Logs sync marker updated to %s", newest[0]["created_at"])

        return {
            "status": "success",
            "rows_written": len(values) - 1,
            "range": sheet_range,
            "feedback_rows_written": len(feedback_values) - 1 if update_feedback else 0,
            "log_rows_written": logs_written,
            "logs_mode": logs_mode,
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": f"sheet_update_failed:{e}",
        }


def send_discord_notification(message: str, embed: Optional[dict] = None):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        return {"status": "skipped", "reason": "missing_webhook"}

    payload = {"content": message}
    if embed:
        payload["embeds"] = [embed]

    try:
        requests.post(webhook_url, json=payload, timeout=10)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "reason": f"discord_send_failed:{e}"}


def post_daily_leaderboard(db_path: str = "competition.db"):
    ranking = RankingSystem(db_path=db_path)
    leaderboard = ranking.get_leaderboard(include_baseline=True)
    if not leaderboard:
        return {"status": "skipped", "reason": "empty_leaderboard"}

    embed = {"title": "Bomberland Daily Leaderboard", "color": 16766720, "fields": []}
    for i, row in enumerate(leaderboard[:5], start=1):
        embed["fields"].append(
            {
                "name": f"#{i} {row['team_name']}",
                "value": (
                    f"Score: {row['score']:.2f} | "
                    f"mu={row['mu']:.2f}, sigma={row['sigma']:.2f} | "
                    f"games={row['n_games']}"
                ),
                "inline": False,
            }
        )

    return send_discord_notification("Latest leaderboard update", embed=embed)


if __name__ == "__main__":
    result = update_google_sheets(db_path="competition.db")
    print(result)
