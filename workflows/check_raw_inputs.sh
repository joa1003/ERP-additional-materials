#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$ERP_PROJECT_ROOT"

required=(
  "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv"
  "data/raw/LCL/csv/data_collection/data_tables/survey_questions.csv"
  "data/raw/LCL/csv/data_collection/data_tables/survey_answers.csv"
  "data/raw/ISSDA_CER/File1.txt"
  "data/raw/ISSDA_CER/File2.txt"
  "data/raw/ISSDA_CER/File3.txt"
  "data/raw/ISSDA_CER/File4.txt"
  "data/raw/ISSDA_CER/File5.txt"
  "data/raw/ISSDA_CER/File6.txt"
  "data/raw/ISSDA_CER/Smart meters Residential pre-trial survey data.csv"
  "data/raw/ISSDA_CER/SurveySourceData_survey_original/RESIDENTIAL PRE TRIAL SURVEY.pdf"
  "data/raw/ISSDA_CER/SurveySourceData_survey_original/SME and Residential allocations.xlsx"
  "data/raw/ISSDA_CER/SurveySourceData_survey_original/File Manifest - Smart Meter Electricity Trial Data v1.0.1.pdf"
)

for path in "${required[@]}"; do
  require_file "$ERP_PROJECT_ROOT/$path"
done

echo "RAW INPUT CHECK: PASS (${#required[@]} required files)"
