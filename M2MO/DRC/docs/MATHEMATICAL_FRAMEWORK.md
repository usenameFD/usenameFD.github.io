# Mathematical Framework Reference

## Complete Model Specification

### Probability Space
(Ω, ℱ, ℙ) with:
- Ω = all states at horizon T=1 year
- ℱ = information at T
- ℙ = historical (real-world) measure

### Latent Variable Model
For each asset i:
```
X_i = β_i^T F + σ_i ε_i
```

Where:
- F ~ N(0, I_K): systematic factors
- ε_i ~ N(0,1): idiosyncratic shock
- β_i ∈ ℝ^K: factor loadings
- σ_i = √(1 - ||β_i||²): idiosyncratic volatility

### Default Mechanism
```
D_i = 1{X_i < c_i}
c_i = Φ^{-1}(PD_i)
```

### Portfolio Loss
```
L = Σ_i JTD_i × D_i × LGD_i
```

### DRC (Default Risk Charge)
```
DRC = VaR_{0.999}(L) = inf{ℓ ≥ 0 : P(L ≤ ℓ) ≥ 0.999}
```

## All 7 Corrections

1. **JTD**: Shares × Price(t) [dynamic, not static weights]
2. **LGD**: Equity-specific 60-100% [not bond 40-60%]
3. **Correlation**: EWMA λ=0.94, 120 days [not 5-year rolling]
4. **Idiosyncratic**: eps @ diag(sigma) [not eps * sigma]
5. **Variance**: Validate Var(X_i) = 1 [explicit check]
6. **Governance**: Pre-announced dates [not data-dependent]
7. **Cholesky**: F = Z @ L^T [proper matrix op]

## References

- Vasicek (2002): Latent variable framework
- Laloux et al. (2000): Random matrix theory
- Glasserman & Li (2005): Importance sampling
- CRR3 Article 325bp: Regulatory requirements
