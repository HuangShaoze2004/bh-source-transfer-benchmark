# dd_original sensitivity data

Sensitivity only: Dreher Doyle additive identities remain missing; no additive was inferred or fabricated. These incomplete inputs cannot validate a chemically complete additive-aware predictor.

Completed 30 neural training trajectories, 15 RF fits and the full 75-file prediction matrix. Independent metric reconstruction checked 211,635 prediction rows. The main report includes the native-test results and fixed-truth common-row comparison. Training data, source count and optimizer-update counts change together in this branch. Use the main project's environment and scripts with this directory as the project data root. weights and src are relative symlinks.

Label policy: Yield >= 10, including exactly 10. Split policy: recomputed strict LOSO with identical normalization/qualification/inner-source rules.
