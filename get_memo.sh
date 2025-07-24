#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

awk -F',' '
{
    gsub(/^[ \t]+|[ \t]+$/, "", $4);  # trim whitespace
    mib = $4 + 0;
    if (mib > max) max = mib;
}
END {
    gib = max / 1024;
    printf "Max of column 4: %.2f GiB (%.0f MiB)\n", gib, max;
}
' "$1"

