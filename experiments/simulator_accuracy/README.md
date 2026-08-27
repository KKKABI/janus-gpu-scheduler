# Janus 4.7 simulator-accuracy experiment

This experiment separates the Static/TD feasibility predicates from the
Janus/DRT final candidate scorer.

The discovery phase runs both a Static-reference path and a TD-reference path
for each of seven models.  At every visited scheduler state it applies the
paper controls (all resource-bearing operators are HP; at most six ready GPU
operators, retained by predicted achieved occupancy), enumerates exact groups
of size two through five, and records paired Static/TD yes/no predictions.

`reference_variant` only determines which scheduler states are visited.  It is
not a prediction label and must not enter the precision numerator or
denominator.

The next phase will select or exhaustively retain the union of positive
predictions and validate each exact group with a common-start multi-stream
CUDA Graph plus NSYS kernel-overlap analysis.  No 5/5 stability requirement is
part of the primary Janus 4.7 precision metric.
