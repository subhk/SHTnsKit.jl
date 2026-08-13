#!/bin/sh
set -eu

generator=${1:?usage: check_reproducible.sh GENERATOR}
case "$generator" in
    /*) ;;
    *) generator=$(pwd)/${generator#./} ;;
esac
root=$(mktemp -d "${TMPDIR:-/tmp}/shtns37-repro.XXXXXX")
trap 'rm -rf "$root"' EXIT HUP INT TERM

for run in 1 2 3; do
    "$generator" "$root/$run" >/dev/null
done

for run in 2 3; do
    diff -qr "$root/1" "$root/$run"
done
