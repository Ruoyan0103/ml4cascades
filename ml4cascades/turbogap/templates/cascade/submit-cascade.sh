#!/bin/bash

#SBATCH --time=11:00:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=cas_{num}
#SBATCH --error=job.error
#SBATCH --output=job.output

module load triton/2025.1-gcc
module load openblas/0.3.28
module load openmpi/5.0.3

set -e

for f in $(ls input_s* | sort -V); do
    stage=${f#input_}          # input_s1 -> s1

    cp "$f" input

    before=$(mktemp)
    ls -1 > "$before"

    srun turbogap md

    mkdir -p "$stage"
    after=$(mktemp)
    ls -1 > "$after"

    comm -13 "$before" "$after" | while read -r file; do
        [ -f "$file" ] && mv "$file" "$stage/"
    done
    rm -f "$before" "$after"

    cp "$f" "$stage/input"     # keep a record of the input actually used

    next_num=$(( ${stage#s} + 1 ))
    next_input="input_s${next_num}"

    # turbogap's atoms_file reader always consumes the FIRST frame of an
    # extxyz file, never the last - so the next stage can't just point at
    # this stage's raw (multi-frame) trajectory_out.xyz. Slice out the
    # final frame here instead, named for the stage that will consume it.
    if [ -f "$next_input" ]; then
        n_atoms=$(head -n 1 "$stage/trajectory_out.xyz")
        tail -n "$((n_atoms + 2))" "$stage/trajectory_out.xyz" > "atoms_file_s${next_num}.xyz"
    fi

    # generate the FDM temperature input for the NEXT stage from this
    # stage's last eph-ToutData.txt frame, if a next-stage input exists
    if [ -f "$next_input" ]; then
        read -r gsx gsy gsz _ < <(sed -n '4p' T_input_s1.fdm)
        GRID_POINTS=$((gsx * gsy * gsz))
        FRAME_LINES=$((GRID_POINTS + 1))   # 1 step-count line + data
        {
            head -n 8 T_input_s1.fdm
            tail -n "$FRAME_LINES" "${stage}/eph-ToutData.txt" | tail -n +2
        } > "T_input_s${next_num}.fdm"
    fi
done

