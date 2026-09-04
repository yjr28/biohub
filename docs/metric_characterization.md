# Metric Characterization Results

Verified by executable tests against the pinned organizer evaluator on **2026-09-04**.

Upstream evaluator: `royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2`

This file records observed behavior, not a replacement metric specification. The tests in `tests/test_metric_characterization.py` are the executable contract.

## Passed characterizations

| Property | Observed behavior |
|---|---|
| Perfect linear edge | TP=1, FP=0, FN=0, raw and adjusted edge Jaccard = 1 when predicted count equals estimate |
| 7 µm boundary | exactly 7.0 µm matches; 7.000001 µm does not |
| Anisotropic physical scaling | 4 z-voxels at 1.625 µm/voxel (6.5 µm) matches; 5 z-voxels (8.125 µm) does not |
| Time awareness | co-located nodes at different timepoints do not match |
| Skip edge | `t -> t+2` does not recover the two missing consecutive GT edges |
| Backward edge | backward predicted edge is removed from edge TP/FP accounting |
| Sparse-GT track-end continuation | an edge extending from an annotated terminal node to an unmatched node can be invisible to edge FP accounting |
| Annotated-interior wrong edge | a wrong edge entering an annotated interior node is an edge FP |
| Duplicate edge | duplicate `(source,target)` predictions cannot inflate TP/score |
| Out-degree >2 | edge scorer keeps the two lowest edge IDs; insertion order can therefore change the scored graph |
| Node-count adjustment | overprediction lowers adjusted edge score; underprediction raises it; no upper clamp was observed (`1.05` from raw `1.0` in the controlled case) |
| Exact division | exact local lineage produces division TP=1, FP=0, FN=0 |
| Run aggregation | adjusted edge Jaccard is sample-size weighted, not a plain mean; zero-division split drops the division term |

## Strategic implications we may rely on

1. **Use physical distance, never raw voxel distance, for metric-adjacent analysis.** Z anisotropy is large enough to change match/no-match outcomes.
2. **Consecutive-edge recovery is the primary edge objective.** Skip links are not substitutes in the scorer.
3. **Candidate ordering matters if a node can exceed two outgoing edges.** Any optimizer/adaptor must emit a deliberately ordered/final graph rather than treating edge insertion order as irrelevant serialization detail.
4. **Sparse-label visibility and node-count calibration are separate channels.** An extra prediction can avoid edge FP accounting while still affecting the node-count adjustment.
5. **Do not compare raw edge Jaccard alone.** The competition-style adjusted score can move in a different direction as predicted node count changes.
6. **Per-sample adjustment must occur before run aggregation.** A hand-computed global node penalty is not equivalent to the organizer scoring path.

## What this does *not* establish

- whether deliberately underpredicting nodes improves hidden-test score in practice;
- the best detector threshold;
- the best candidate radius or tracker;
- hidden-test node-count estimates/distribution;
- embryo generalization behavior;
- public/private leaderboard correlation;
- whether any nonstandard graph construction is desirable or robust.

Those are experiment questions, not metric-contract facts.
