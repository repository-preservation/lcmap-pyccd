#!/bin/bash

gprof2dot -f pstats $1 > output.dot
now=`date "+%Y-%m-%d_%H:%M:%S"`
dot output.dot -T svg -o output_${now}.svg

