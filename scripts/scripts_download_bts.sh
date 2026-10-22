#!/usr/bin/env bash
# Download and unpack the pinned BTS monthly files.
set -euo pipefail

base='https://transtats.bts.gov/PREZIP'
zip_dir='data/raw/zips'
csv_dir='data/raw/csv'
mkdir -p "$zip_dir" "$csv_dir"

for year in 2023 2024; do
  for month in {1..12}; do
    file="On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${year}_${month}.zip"
    dest="$zip_dir/$file"

    if [[ ! -f "$dest" ]]; then
      echo "downloading $file"
      curl -L --fail --silent --show-error --retry 3 \
        "$base/$file" -o "$dest"
    else
      echo "exists $file"
    fi

    echo "checking and extracting $file"
    unzip -tq "$dest"
    unzip -oq "$dest" '*.csv' -d "$csv_dir"
  done
done
echo "download and extraction complete"
