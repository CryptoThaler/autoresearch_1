# autoresearch-envsci

This is an experiment to have the LLM do its own research on environmental
science prediction models — specifically crop yield forecasting using fused
NASA satellite observations and USDA agricultural ground truth data.

## Domain context

The goal is to build a neural network that predicts county-level crop yield
(bushels/acre) from satellite-derived environmental features. This is a
real and active area of applied research at the intersection of remote sensing,
agroclimatology, and food security.

The model ingests time-series environmental features for a growing season
(temperature, precipitation, solar radiation, vegetation indices, soil moisture)
and outputs a single scalar prediction: end-of-season yield for a given
crop in a given county.

Key domain knowledge for the agent:

- Crop yield is driven by accumulated growing conditions, not snapshots.
-   Temporal modeling matters — a drought in July has very different impact
-     than a drought in April.
- - NDVI (Normalized Difference Vegetation Index) is the single strongest
  -   satellite predictor of yield, but it saturates at high biomass. EVI
  -     (Enhanced Vegetation Index) handles this better for dense canopy crops
  -   like corn.
  -   - Soil moisture is critical but noisy from satellite retrievals. NASA POWER
      -   provides modeled soil moisture proxies through precipitation and humidity.
      -   - The relationship between temperature and yield is nonlinear —
          -   moderate warmth helps, extreme heat (>35C) causes pollen sterility
          -     in corn and dramatic yield collapse.
          - - Growing Degree Days (GDD) and Killing Degree Days (KDD) are standard
            -   agronomic features that compress daily temperature into biologically
            -     meaningful accumulations.
            - - County-level yield data from USDA NASS is the ground truth. It is
              -   annual, covers all major US crops, and goes back decades but has
              -     reporting lags and occasional revisions.
             
              - ## Data sources (reference only — already downloaded by prepare.py)
             
              - NASA sources:
              - - NASA POWER (power.larc.nasa.gov) — daily gridded solar, temperature,
                -   precipitation, humidity, wind at 0.5 degree resolution. API-accessible.
                -     Parameters: T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN,
                -   RH2M, WS2M, T2MDEW.
                -   - NASA LP DAAC / MODIS (earthdata.nasa.gov) — 16-day composite NDVI and
                    -   EVI at 250m resolution (MOD13Q1 product). LAI/FPAR (MOD15A2H).
                    -     Land surface temperature (MOD11A2).
                 
                    - USDA sources:
                    - - USDA NASS Quick Stats (quickstats.nass.usda.gov) — county-level annual
                      -   crop yield, production, planted/harvested acreage for all major
                      -     commodities. This is the ground truth label.
                      - - USDA NASS datasets (nass.usda.gov/datasets/) — bulk downloads including
                        -   qs.crops, qs.environmental. Pre-filtered for corn, soybean, winter wheat.
                       
                        -   All data has been pre-processed by prepare.py into county-level growing
                        -   season feature tensors and stored in ~/.cache/autoresearch-envsci/.
                       
                        -   ## Setup
                       
                        -   To set up a new experiment, work with the user to:
                       
                        -   1. **Agree on a run tag**: propose a tag based on today's date (e.g.
                            2.    mar8). The branch autoresearch/<tag> must not already exist —
                            3.   this is a fresh run.
                         
                            4.   2. **Create the branch**: git checkout -b autoresearch/<tag> from
                                 3.    current master.
                              
                                 4.3. **Read the in-scope files**: The repo is small. Read these files for
                                    full context:
                                    - README.md — repository context and background on the problem.
                                    -    - prepare.py — fixed constants, data download/preprocessing,
                                         -      county feature extraction, train/val/test splits, evaluation
                                         -       harness. Do not modify.
                                         -      - train.py — the file you modify. Model architecture, optimizer,
                                         -       loss function, training loop.
                                     
                                         -   4. **Verify data exists**: Check that ~/.cache/autoresearch-envsci/
                                             5.    contains processed feature tensors (train_features.pt,
                                             6.   train_labels.pt, val_features.pt, val_labels.pt) and metadata
                                             7.      (counties.json, feature_names.json). If not, tell the human to
                                             8.     run: uv run prepare.py
                                          
                                             9. 5. **Initialize results.tsv**: Create results.tsv with header row and
                                                6.    baseline entry. The baseline results are already known from the
                                                7.   output format section below. Do NOT re-run the baseline — just
                                                8.      record it.
                                               
                                                9.  6. **Confirm and go**: Confirm setup looks good. Once you get
                                                    7.    confirmation, kick off the experimentation.
                                                  
                                                    8.## Experimentation

                                                    Each experiment runs on a single GPU. The training script runs for a
                                                    **fixed time budget of 5 minutes** (wall clock training time, excluding
                                                    startup). You launch it simply as: uv run train.py

                                                    **What you CAN do:**
                                                    - Modify train.py — this is the only file you edit. Everything is
                                                    -   fair game: model architecture, optimizer, hyperparameters, loss
                                                    -     function, feature engineering within the model (learned temporal
                                                    -   aggregations, attention over time steps, cross-feature interactions),
                                                    -     batch size, model size, regularization, data augmentation, etc.
                                                  
                                                    - **What you CANNOT do:**
                                                    - - Modify prepare.py. It is read-only. It contains the fixed
                                                      -   evaluation, data loading, feature extraction, and training constants
                                                      -     (time budget, train/val splits, etc).
                                                      - - Install new packages or add dependencies. You can only use what is
                                                        -   already in pyproject.toml.
                                                        -   - Modify the evaluation harness. The evaluate_rmse and evaluate_r2
                                                            -   functions in prepare.py are the ground truth metrics.
                                                         
                                                            -   **The goal is simple: get the lowest val_rmse (root mean squared error
                                                            -   in bushels/acre on the held-out validation counties/years).**
                                                         
                                                            -   Since the time budget is fixed, you do not need to worry about training
                                                            -   time — it is always 5 minutes. Everything is fair game: change the
                                                            -   architecture, the optimizer, the hyperparameters, the loss function,
                                                            -   the way the model processes temporal features. The only constraint is
                                                            -   that the code runs without crashing and finishes within the time budget.
                                                         
                                                            -   **VRAM** is a soft constraint. Some increase is acceptable for meaningful
                                                            -   val_rmse gains, but it should not blow up dramatically.
                                                         
                                                            -   **Simplicity criterion**: All else being equal, simpler is better. A
                                                            -   small improvement that adds ugly complexity is not worth it. Conversely,
                                                            -   removing something and getting equal or better results is a great outcome
                                                            -   — that is a simplification win. When evaluating whether to keep a change,
                                                            -   weigh the complexity cost against the improvement magnitude. A 0.1
                                                            -   bu/acre val_rmse improvement that adds 30 lines of hacky code? Probably
                                                            -   not worth it. A 0.1 bu/acre improvement from deleting code? Definitely
                                                            -   keep. An improvement of ~0 but much simpler code? Keep.
                                                         
                                                            -   **Domain-aware experimentation guidelines:**
                                                         
                                                            -   Promising directions to explore (ordered roughly by expected impact):
                                                            -   - Temporal modeling: the baseline uses simple feature averaging across
                                                                -   the growing season. Try temporal attention, 1D convolutions over the
                                                                -     biweekly time steps, LSTMs, or learned phenological phase weighting
                                                                -   (early season vs mid-season vs grain fill).
                                                                -   - Nonlinear temperature response: the relationship between temperature
                                                                    -   and yield is not linear. Piecewise linear embeddings, learned
                                                                    -     threshold functions, or explicit GDD/KDD feature engineering inside
                                                                    -   the model could capture the heat stress cliff.
                                                                    -   - Multi-task learning: predicting yield for corn, soybean, and wheat
                                                                        -   simultaneously with shared representations may improve generalization
                                                                        -     especially for counties with sparse data.
                                                                        - - Feature interaction layers: soil moisture x temperature interactions
                                                                          -   are agronomically meaningful (drought stress is worse in heat). Let
                                                                          -     the model learn cross-feature interactions explicitly.
                                                                          - - Loss function: Huber loss or quantile regression may be more robust
                                                                            -   than MSE given yield distribution skew and occasional extreme events.
                                                                            -   - Regularization: yield data is relatively small (~3000 county-year
                                                                                -   observations per crop). Dropout, weight decay, and early stopping
                                                                                -     matter more here than in LLM training.
                                                                             
                                                                                - Directions that sound good but are usually dead ends:
                                                                                - - Very deep models. The dataset is small. Models with more than 4-6
                                                                                  -   layers tend to overfit aggressively. Regularization is more valuable
                                                                                  -     than depth.
                                                                                  - - Complex attention over the spatial dimension with full county-to-county
                                                                                    -   attention. The county graph is large and sparse. Simple spatial
                                                                                    -     features (state means, neighbor averages) work better for the compute
                                                                                    -   budget.
                                                                                    -   - Trying to predict daily yield — the label is annual. Sub-seasonal
                                                                                        -   auxiliary targets can help but daily predictions are noise.
                                                                                     
                                                                                        -   **The first run**: Your very first run should always be to establish the
                                                                                        -   baseline, so you will run the training script as is.
                                                                                     
                                                                                        -   ## Output format
                                                                                     
                                                                                        -   Once the script finishes it prints a summary like this:
                                                                                     
                                                                                        -   ```
                                                                                            ---
                                                                                            val_rmse: 14.230000
                                                                                            val_r2: 0.782000
                                                                                            val_mae: 10.450000
                                                                                            training_seconds: 300.1
                                                                                            total_seconds: 318.4
                                                                                            peak_vram_mb: 8240.5
                                                                                            num_steps: 1200
                                                                                            num_params_K: 485.2
                                                                                            crop: corn
                                                                                            num_counties: 2847
                                                                                            num_years: 15
                                                                                            ```

                                                                                            You can extract the key metric from the log file:

                                                                                            ```
                                                                                            grep "^val_rmse:" run.log
                                                                                            ```

                                                                                            ## Logging results

                                                                                            When an experiment is done, log it to results.tsv (tab-separated, NOT
                                                                                            comma-separated — commas break in descriptions). The TSV has a header
                                                                                            row and 6 columns:

                                                                                            ```
                                                                                            commit	val_rmse	val_r2	memory_gb	status	description
                                                                                            ```

                                                                                            1. git commit hash (short, 7 chars)
                                                                                            2. 2. val_rmse achieved (e.g. 14.230000) — use 0.000000 for crashes
                                                                                               3. 3. val_r2 achieved (e.g. 0.782000) — use 0.000000 for crashes
                                                                                                  4. 4. peak memory in GB, round to .1f — use 0.0 for crashes
                                                                                                     5. 5. status: keep, discard, or crash
                                                                                                        6. 6. short text description of what this experiment tried
                                                                                                          
                                                                                                           7. Example:
                                                                                                          
                                                                                                           8. ```
                                                                                                              commit	val_rmse	val_r2	memory_gb	status	description
                                                                                                              a1b2c3d	14.230000	0.782000	8.1	keep	baseline MLP with season-averaged features
                                                                                                              b2c3d4e	13.870000	0.794000	8.3	keep	add temporal attention over biweekly steps
                                                                                                              c3d4e5f	14.510000	0.770000	8.1	discard	replace ReLU with GELU in MLP layers
                                                                                                              d4e5f6g	0.000000	0.000000	0.0	crash	LSTM with 512 hidden (OOM)
                                                                                                              e5f6g7h	13.620000	0.801000	9.2	keep	add GDD/KDD piecewise temperature encoding
                                                                                                              f6g7h8i	13.580000	0.803000	9.1	keep	Huber loss delta=10 replaces MSE
                                                                                                              ```
                                                                                                              
                                                                                                              ## The experiment loop
                                                                                                              
                                                                                                              The experiment runs on a dedicated branch (e.g. autoresearch/mar8).
                                                                                                              
                                                                                                              LOOP FOREVER:
                                                                                                              
                                                                                                              1. Look at the git state: the current branch/commit we are on
                                                                                                              2. 2. Tune train.py with an experimental idea by directly hacking the code
                                                                                                                 3. 3. git commit
                                                                                                                    4. 4. Run the experiment: uv run train.py > run.log 2>&1
                                                                                                                       5. 5. Read out the results: grep "^val_rmse:\|^val_r2:\|^peak_vram_mb:" run.log
                                                                                                                          6. 6. If the grep output is empty, the run crashed. Run tail -n 50 run.log
                                                                                                                             7.    to read the Python stack trace and attempt a fix. If you cannot get
                                                                                                                             8.   things to work after more than a few attempts, give up.
                                                                                                                             9.   7. Record the results in the tsv
                                                                                                                                  8. 8. If val_rmse improved (lower), you advance the branch keeping the commit
                                                                                                                                     9. 9. If val_rmse is equal or worse, you git reset back to where you started
                                                                                                                                       
                                                                                                                                        10. The idea is that you are a completely autonomous researcher trying things
                                                                                                                                        11. out. If they work, keep. If they do not, discard. And you are advancing
                                                                                                                                        12. the branch so that you can iterate.
                                                                                                                                       
                                                                                                                                        13. **Timeout**: Each experiment should take ~5 minutes total (+ a few
                                                                                                                                        14. seconds for startup and eval overhead). If a run exceeds 10 minutes,
                                                                                                                                        15. kill it and treat it as a failure (discard and revert).
                                                                                                                                       
                                                                                                                                        16. **Crashes**: If a run crashes (OOM, or a bug, or etc.), use your
                                                                                                                                        17. judgment: If it is something dumb and easy to fix (a typo, a missing
                                                                                                                                        18. import), fix it and re-run. If the idea itself is fundamentally broken,
                                                                                                                                        19. just skip it, log crash as the status in the tsv, and move on.
                                                                                                                                       
                                                                                                                                        20. **NEVER STOP**: Once the experiment loop has begun (after the initial
                                                                                                                                        21. setup), do NOT pause to ask the human if you should continue. Do NOT ask
                                                                                                                                        22. should I keep going or is this a good stopping point. The human might be
                                                                                                                                        23. asleep, or gone from a computer and expects you to continue working
                                                                                                                                        24. indefinitely until you are manually stopped. You are autonomous. If you
                                                                                                                                        25. run out of ideas, think harder — re-read the domain context for new
                                                                                                                                        26. angles, try combining previous near-misses, explore more radical
                                                                                                                                        27. architectural changes, think about agronomic interactions you have not
                                                                                                                                        28. modeled yet. The loop runs until the human interrupts you, period.
