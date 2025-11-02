"""
Google Sheets logger for AISAI experiment results.
Logs each trial result in real-time to Google Sheets.
"""

import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class SheetsLogger:
    """Logger for experiment results to Google Sheets."""

    def __init__(self, credentials_path: Optional[str] = None, spreadsheet_name: Optional[str] = None, spreadsheet_id: Optional[str] = None):
        """
        Initialize Google Sheets logger.

        Args:
            credentials_path: Path to service account JSON file
            spreadsheet_name: Name of the spreadsheet to use
            spreadsheet_id: ID of the spreadsheet to use (takes precedence over name)
        """
        self.credentials_path = credentials_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS")
        self.spreadsheet_name = spreadsheet_name or os.getenv("SPREADSHEET_NAME", "AISAI_Experiment_Results")
        self.spreadsheet_id = spreadsheet_id or os.getenv("SPREADSHEET_ID")

        if not self.credentials_path:
            raise ValueError("Google Sheets credentials path not provided")

        # Setup credentials and authorize
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_path, scope
            )
            self.client = gspread.authorize(creds)
            print(f"✓ Google Sheets client authorized")

            # Open or create spreadsheet
            try:
                if self.spreadsheet_id:
                    # Use spreadsheet ID if provided
                    self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                    print(f"✓ Opened spreadsheet by ID: {self.spreadsheet_id}")
                else:
                    # Fall back to opening by name
                    self.spreadsheet = self.client.open(self.spreadsheet_name)
                    print(f"✓ Opened existing spreadsheet: {self.spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                if self.spreadsheet_id:
                    raise ValueError(f"Spreadsheet with ID {self.spreadsheet_id} not found or not accessible")
                self.spreadsheet = self.client.create(self.spreadsheet_name)
                print(f"✓ Created new spreadsheet: {self.spreadsheet_name}")

            # Initialize worksheets
            self._initialize_worksheets()

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}\n"
                "Please download service account JSON from Google Cloud Console."
            )

    def _initialize_worksheets(self):
        """Create or get worksheets for each prompt type."""
        worksheet_names = ["A_baseline", "B_against_ai", "C_against_self", "summary"]

        for name in worksheet_names:
            try:
                worksheet = self.spreadsheet.worksheet(name)
                print(f"✓ Found existing worksheet: {name}")
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title=name, rows=1000, cols=20)
                print(f"✓ Created new worksheet: {name}")

                # Add headers
                if name != "summary":
                    headers = [
                        "Timestamp",
                        "Trial",
                        "Provider",
                        "Model",
                        "Prompt_Type",
                        "Temperature",
                        "Reasoning_Config",
                        "Valid_JSON",
                        "Guess",
                        "Reasoning",
                        "Raw_Response",
                        "API_Full_Response",
                        "Success",
                        "Error",
                        "Input_Tokens",
                        "Output_Tokens",
                        "Total_Tokens"
                    ]
                else:
                    headers = [
                        "Prompt_Type",
                        "Provider",
                        "Model",
                        "Total_Trials",
                        "Successful_Trials",
                        "Failed_Trials",
                        "Mean_Guess",
                        "Median_Guess",
                        "Std_Guess",
                        "Min_Guess",
                        "Max_Guess"
                    ]
                worksheet.append_row(headers)

    def _truncate_field(self, text: str, max_length: int = 49000) -> str:
        """
        Truncate text to fit Google Sheets cell limit (50,000 chars).

        Args:
            text: Text to truncate
            max_length: Maximum length (default 49000 to leave margin)

        Returns:
            Truncated text with indicator if truncated
        """
        if not text:
            return ""

        text_str = str(text)
        if len(text_str) <= max_length:
            return text_str

        truncated = text_str[:max_length - 100]
        truncated += "\n\n[... TRUNCATED - exceeded 50k char limit ...]"
        return truncated

    def log_trial(self, trial_data: Dict[str, Any]):
        """
        Log a single trial result to the appropriate worksheet.

        Args:
            trial_data: Dictionary containing trial information
        """
        prompt_type = trial_data.get("prompt_type", "A_baseline")

        try:
            worksheet = self.spreadsheet.worksheet(prompt_type)
        except gspread.WorksheetNotFound:
            print(f"Warning: Worksheet {prompt_type} not found, using A_baseline")
            worksheet = self.spreadsheet.worksheet("A_baseline")

        # Parse response
        parsed = trial_data.get("parsed_response", {})
        valid_json = parsed is not None and isinstance(parsed, dict)
        guess = parsed.get("guess", "") if parsed else ""
        reasoning = parsed.get("reasoning", "") if parsed else ""

        # Fallback: Try to extract guess and reasoning from raw response if parsing failed
        if not valid_json or not guess:
            raw_response = trial_data.get("raw_response", "")
            if raw_response:
                import re

                # Try to extract guess number
                if not guess:
                    # Look for "guess": number patterns
                    guess_patterns = [
                        r'"guess"\s*:\s*(\d+\.?\d*)',
                        r"'guess'\s*:\s*(\d+\.?\d*)",
                        r'guess\s*[:=]\s*(\d+\.?\d*)',
                        r'I (?:will )?guess\s+(\d+\.?\d*)',
                        r'my guess is\s+(\d+\.?\d*)',
                    ]
                    for pattern in guess_patterns:
                        match = re.search(pattern, raw_response, re.IGNORECASE)
                        if match:
                            guess = float(match.group(1))
                            break

                # Try to extract reasoning
                if not reasoning:
                    # Look for "reasoning": "text" patterns
                    reasoning_patterns = [
                        r'"reasoning"\s*:\s*"([^"]+)"',
                        r"'reasoning'\s*:\s*'([^']+)'",
                        r'"reasoning"\s*:\s*\[([^\]]+)\]',
                    ]
                    for pattern in reasoning_patterns:
                        match = re.search(pattern, raw_response, re.DOTALL)
                        if match:
                            reasoning = match.group(1)
                            break

        # Handle reasoning as list or string
        if isinstance(reasoning, list):
            reasoning = "\n".join(str(item) for item in reasoning)
        elif reasoning:
            reasoning = str(reasoning)
        else:
            reasoning = ""

        # Prepare row data (truncate api_full_response to avoid Google Sheets 50k char limit)
        row = [
            datetime.now().isoformat(),
            trial_data.get("trial_number", ""),
            trial_data.get("provider", ""),
            trial_data.get("model", ""),
            prompt_type,
            trial_data.get("temperature", ""),
            trial_data.get("reasoning_config", ""),
            valid_json,
            guess,
            reasoning,
            trial_data.get("raw_response", ""),
            self._truncate_field(trial_data.get("api_full_response", "")),  # Can exceed 50k with extended thinking
            trial_data.get("success", False),
            trial_data.get("error", ""),
            trial_data.get("usage", {}).get("input_tokens", ""),
            trial_data.get("usage", {}).get("output_tokens", ""),
            trial_data.get("usage", {}).get("total_tokens", "")
        ]

        # Append row
        try:
            worksheet.append_row(row)
            print(f"✓ Logged trial {trial_data.get('trial_number')} for {trial_data.get('model')} - {prompt_type}")
        except Exception as e:
            print(f"✗ Error logging trial: {e}")

    def get_worksheet_data(self, worksheet_name: str):
        """Get all data from a worksheet."""
        try:
            worksheet = self.spreadsheet.worksheet(worksheet_name)
            return worksheet.get_all_records()
        except gspread.WorksheetNotFound:
            print(f"Worksheet {worksheet_name} not found")
            return []

    def update_summary(self, summary_data: Dict[str, Any]):
        """Update the summary worksheet with aggregated statistics."""
        try:
            worksheet = self.spreadsheet.worksheet("summary")

            row = [
                summary_data.get("prompt_type", ""),
                summary_data.get("provider", ""),
                summary_data.get("model", ""),
                summary_data.get("total_trials", 0),
                summary_data.get("successful_trials", 0),
                summary_data.get("failed_trials", 0),
                summary_data.get("mean_guess", ""),
                summary_data.get("median_guess", ""),
                summary_data.get("std_guess", ""),
                summary_data.get("min_guess", ""),
                summary_data.get("max_guess", "")
            ]

            worksheet.append_row(row)
            print(f"✓ Updated summary for {summary_data.get('model')} - {summary_data.get('prompt_type')}")
        except Exception as e:
            print(f"✗ Error updating summary: {e}")

    def clear_worksheet(self, worksheet_name: str):
        """Clear all data from a worksheet (keeping headers)."""
        try:
            worksheet = self.spreadsheet.worksheet(worksheet_name)
            # Get all values
            all_values = worksheet.get_all_values()
            if len(all_values) > 1:
                # Keep header row, delete rest
                worksheet.delete_rows(2, len(all_values))
                print(f"✓ Cleared worksheet: {worksheet_name}")
        except Exception as e:
            print(f"✗ Error clearing worksheet: {e}")


def create_service_account_instructions():
    """Print instructions for creating a service account."""
    instructions = """
    To use Google Sheets integration, you need to create a service account:

    1. Go to Google Cloud Console: https://console.cloud.google.com/
    2. Create a new project (or select existing)
    3. Enable Google Sheets API and Google Drive API
    4. Go to "Credentials" → "Create Credentials" → "Service Account"
    5. Create a service account and download the JSON key file
    6. Save the JSON file to: credentials/service_account.json
    7. Share your Google Sheet with the service account email (found in JSON)

    The service account email looks like:
    your-service-account@your-project.iam.gserviceaccount.com

    Share the spreadsheet with this email with "Editor" permissions.
    """
    print(instructions)


if __name__ == "__main__":
    # Test the logger
    print("\n" + "="*50)
    print("Testing Google Sheets Logger")
    print("="*50)

    # Check if credentials exist
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "credentials/service_account.json")

    if not os.path.exists(creds_path):
        print(f"\n✗ Credentials file not found: {creds_path}")
        create_service_account_instructions()
    else:
        try:
            logger = SheetsLogger()
            print("\n✓ Google Sheets logger initialized successfully!")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            create_service_account_instructions()
