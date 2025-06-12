#!/bin/bash
columns=$3
columns="0,5,7-8,10-14"

csv2md -C $columns $1 | sed 's/|/ \& /g' | sed 's/^/ /' | sed 's/$/ \\\\/' 

