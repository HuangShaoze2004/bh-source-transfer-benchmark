# Execution plan

Status: CONTINUE. User approved implementation on 2026-11-10.

1. Establish isolated environment and immutable provenance; preserve USER_REQUEST.md.
2. Audit CSV, structures, labels, duplicate/conflicting inputs, descriptors, and input permissions. Exclude Dreher Doyle unless complete additive identity is recovered before freeze.
3. Recompute lawful fingerprints and descriptors; produce strict source-held-out, individually unseen-aryl and unseen-amine splits. No outer performance during development.
4. Validate pretrained encoder training, gradients, throughput/memory, token coverage, GroupDRO arithmetic, SWA and resumption; estimate full cost before choosing fixed update budget.
5. Freeze candidates, split hashes, tuning rules, metrics, seeds and comparisons. Independently tune within each outer fold without cross-fold adaptation.
6. Run R0_RF, R0_HGB, R1_substitute and three neural trajectories (ERM, balanced ERM, GroupDRO), each yielding ordinary and SWA models; three seeds 17,42,2026.
7. Complete all predictions before uniform external evaluation; paired pair-cluster bootstrap and shared-substrate sensitivity; independently reconstruct metrics.
8. Deliver experiment report, audit, artifacts, failures, costs and reproduction commands. No paper writing or submission.

Budgets: <=160 GPU h total; development <=50 GPU h including first feasibility <=12; reserve 80 for formal work, 30 for sensitivity/review/retries. <=500 CPU core h including failed work. Shared total CPU threads <=4, exclusive <=16. GPU jobs serial; wait for other users' jobs.

Implementation tasks are delegated by module; only the root runner may launch GPU jobs. All active workers currently use one CPU/BLAS thread each.

## Verified continuation, 2026-11-10T03:24:13.504935+00:00
- Controller PID712160 and coordinator PID706992 both revalidated live; current neural worker PID720305 was running with increasing CPU time and five completed epochs. Three inner trajectories completed. No duplicate restart.
- Frozen artifact hashes all match. External metrics remain sealed.
- Added and tested post-release audit of real optimizer budgets, training-only checkpoint IDs, full resume state, sample exposure, normalized group weights/ESS and genuine SWA averaging. Final report will require this audit.
- Main formal matrix, two training sensitivities, full statistical evaluation and final experiment report are still pending; completion is not claimed.

## CPU baseline overlap, 2026-11-10T03:43:44.174032+00:00
- First matched neural grid configuration completed all6 inner trajectories (3 objectives x2 sources); next frozen grid is running.
- Precompute only the first outer fold's6 already-planned tree inner jobs through the existing pipeline.job/guard API while GPU work proceeds. Same run IDs, specs, output paths and cache identities will be reused by the primary coordinator. No extra candidate or test evaluation.
- Two R0_RF jobs completed on18231/11107 training rows with2684/8448 internal-validation predictions, all finite. R0_HGB live on CPU slot1 while the sole GPU job uses CPU slot0.
