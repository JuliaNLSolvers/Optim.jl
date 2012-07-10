#!/bin/bash

julia benchmarks/timing.jl > benchmarks/results.tsv
Rscript benchmarks/plot_results.R
open benchmarks/graphs
