#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$ERP_PROJECT_ROOT"

mkdir -p data/raw/LCL/csv/data_collection/data_tables
mkdir -p data/raw/ISSDA_CER/SurveySourceData_survey_original

usage() {
  cat <<'EOF'
Usage:
  bash workflows/prepare_raw_inputs.sh
  bash workflows/prepare_raw_inputs.sh --source-root /path/to/raw

Without --source-root, the script creates the required raw-data directories and
prints the expected files. It never searches the user's computer.

With --source-root, the source root must already contain the documented LCL/ and
ISSDA_CER/ raw-data tree. Only the 13 required provider files are copied.
EOF
}

source_root=""
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--source-root" && $# -eq 2 ]]; then
    source_root="$2"
  else
    usage
    exit 2
  fi
fi

required=(
  "LCL/csv/data_collection/data_tables/consumption_n.csv"
  "LCL/csv/data_collection/data_tables/survey_questions.csv"
  "LCL/csv/data_collection/data_tables/survey_answers.csv"
  "ISSDA_CER/File1.txt"
  "ISSDA_CER/File2.txt"
  "ISSDA_CER/File3.txt"
  "ISSDA_CER/File4.txt"
  "ISSDA_CER/File5.txt"
  "ISSDA_CER/File6.txt"
  "ISSDA_CER/Smart meters Residential pre-trial survey data.csv"
  "ISSDA_CER/SurveySourceData_survey_original/RESIDENTIAL PRE TRIAL SURVEY.pdf"
  "ISSDA_CER/SurveySourceData_survey_original/SME and Residential allocations.xlsx"
  "ISSDA_CER/SurveySourceData_survey_original/File Manifest - Smart Meter Electricity Trial Data v1.0.1.pdf"
)

if [[ -n "$source_root" ]]; then
  source_root="$(cd "$source_root" && pwd)"
  echo "Copying required provider files from: $source_root"
  for rel in "${required[@]}"; do
    src="$source_root/$rel"
    dst="$ERP_PROJECT_ROOT/data/raw/$rel"
    [[ -f "$src" ]] || fail "Required source file is missing: $src"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "  copied: $rel"
  done
fi

echo
echo "Required raw-data root: $ERP_PROJECT_ROOT/data/raw"
echo "Expected files:"
for rel in "${required[@]}"; do
  target="$ERP_PROJECT_ROOT/data/raw/$rel"
  if [[ -s "$target" ]]; then
    printf '  [OK]      %s\n' "$rel"
  else
    printf '  [MISSING] %s\n' "$rel"
  fi
done

echo
echo "Disk requirement: at least 25 GiB free on the output filesystem; 30 GiB or more is recommended."

echo
if bash workflows/check_raw_inputs.sh >/dev/null 2>&1; then
  echo "RAW INPUT PREPARATION: PASS"
else
  echo "RAW INPUT PREPARATION: directories are ready; place the missing provider files shown above, then run:"
  echo "  bash workflows/check_raw_inputs.sh"
fi
