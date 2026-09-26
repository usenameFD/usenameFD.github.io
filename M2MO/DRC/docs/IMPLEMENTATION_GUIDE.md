# Step-by-Step Implementation Guide

## Phase 1: Data Preparation (Week 1)

1. Collect 2-5 years of daily return data
2. Identify assets and their ratings
3. Obtain PD tables by rating
4. Obtain LGD equity table

## Phase 2: Calibration (Week 2)

1. Standardize returns
2. Compute correlation matrix
3. Apply RMT cleaning
4. Extract PCA factors (K≈5-7)
5. Regress returns on PCA scores → β_i
6. Compute σ_i = √(1 - ||β||²)
7. Validate Var(X_i) = 1

## Phase 3: Simulation (Week 3)

1. Set up MC parameters (M=50,000)
2. For each simulation:
   - Draw F ~ N(0, I_K)
   - Draw ε ~ N(0, I_n)
   - Compute X_i = β^T F + σ ε
   - Determine D_i = 1{X_i < c_i}
   - Compute L = Σ JTD × D × LGD
3. Compute VaR_{99.9%}(L)

## Phase 4: Governance & Deployment (Week 4)

1. Set up pre-announced computation dates
2. Document methodology
3. Set up monitoring/alerts
4. Train team
5. Go live
