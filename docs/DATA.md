# Match data

The bundled workbook is named FCF_Futsal_BCN_Gr10_2025_26_DB.xlsx and represents the project's FCF Barcelona Group 10 history snapshot. The repository does not record the exact collection URL, retrieval date or transformation log; these remain provenance gaps. The filename alone should not be treated as proof of complete season coverage.

The CLI records actual completed-match date coverage and a SHA-256 fingerprint in run_manifest.json. It does not fetch or update results. Verify updates against official FCF results before replacing the snapshot.

| Column | Required | Meaning |
|---|---|---|
| Date | Yes | Played date, Excel date or ISO date in CSV |
| HomeTeam / AwayTeam | Yes | Consistent team identifiers |
| HomeGoals / AwayGoals | Yes | Non-negative integer final scores; both blank means unplayed |
| Season | Optional | Non-null season identifier; changes trigger rating reversion |
| Jornada | Optional | Matchweek metadata; chronology uses Date |
| Other columns | No | Preserved during loading, ignored by the model |

Use actual played dates for postponed fixtures. Date ties share the same pre-date information set. Avoid inconsistent team aliases across seasons and duplicate Date/HomeTeam/AwayTeam combinations.

The MIT code license does not grant rights to third-party match data. No claim of an independent data redistribution license is made. For broader publication or commercial reuse, verify the original source's terms.
