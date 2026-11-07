# Contributing

Install requirements-dev.txt and run python -m pytest -q before opening a pull request. Run the CLI with the bundled workbook after changes to loading, modeling or exports.

Keep forecasts chronological. Any new learned preprocessing, tuning or calibration must use only information available before the evaluated match date. Include a regression test when changing evaluation boundaries.

Describe the change, its motivation and validation in the pull request. Include data provenance for new datasets and avoid committing generated out/ files, credentials or personal information.
