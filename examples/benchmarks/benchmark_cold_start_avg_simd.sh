#!/usr/bin/env bash

# This file benchmarks averaged one-shot
# cold startup performance for hotloop_benchmark_simd

set -e

RUNS=1000
TMPFILE=$(mktemp)

# Run benchmark RUNS times, collect outputs
for i in $(seq 1 $RUNS); do
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example hotloop_benchmark_simd >> "$TMPFILE"
done

echo "Averaged Results from $RUNS runs:"
echo "---------------------------------"

awk -F, '
function to_us(val, unit) {
    if (unit == "ns") return val/1000
    if (unit == "µs" || unit == "us") return val
    if (unit == "ms") return val*1000
    if (unit == "s") return val*1000000
    return 0
}
{
    # Only process lines with a duration (comma and time at end)
    if (NF < 2) next
    metric = $1
    # Strip sum and value: keep everything up to "sum = ..."
    sub(/sum = [0-9.eE+-]+/, "", metric)
    # Get time: strip spaces, and extract number/unit
    gsub(/^ +| +$/, "", $2)
    if (match($2, /([0-9.]+)(ns|µs|us|ms|s)/, arr)) {
        tval = arr[1] + 0
        tunit = arr[2]
        tus = to_us(tval, tunit)
        name = metric
        count[name]++
        total[name] += tus
    }
}
END {
    for (k in total) {
        avg = total[k] / count[k]
        printf "%-45s avg = %.3f µs (n=%d)\n", k, avg, count[k]
    }
}
' "$TMPFILE" | sort

rm "$TMPFILE"



