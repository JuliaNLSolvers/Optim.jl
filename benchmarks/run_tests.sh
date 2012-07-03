#!/bin/bash

julia testbed/timing.jl > testbed/results.tsv
Rscript testbed/plot_results.R
open testbed/graphs
