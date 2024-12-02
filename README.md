[![](https://cranlogs.r-pkg.org/badges/SMOTEWB)](https://cran.r-project.org/package=SMOTEWB)

# SMOTEWB

Sağlam and Mehmet’s SMOTE with Boosting (SMOTEWB) (Sağlam and Cengiz
2022) oversampling algorithm for imbalanced data sets. SMOTEWB is noise
resistant and more successful according to Matthew’s Correlation
Coefficient in various data sets. It is a SMOTE-based resampling
technique which creates synthetic data on the links between nearest
neighbors. SMOTEWB uses boosting weights to determine where to generate
new samples and automatically decides the number of neighbors for each
sample. It is robust to noise and outperforms most of the alternatives
according to Matthew Correlation Coefficient metric. Package includes
various resampling methods to be used in imbalanced classification.
Available resampling methods are:

1- ADASYN (Adaptive Synthetic Sampling)

2- BLSMOTE (Borderline SMOTE)

3- GSMOTE (Geometric SMOTE)

4- ROS (Random oversampling)

5- ROSE (Randomly Over Sampling Examples)

6- RSLSMOTE (Relocating safe-level SMOTE with minority outcast handling)

7- RUS (Random undersampling)

8- SLSMOTE (Safe-level Synthetic Minority Oversampling Technique)

9- SMOTE (Synthetic minority oversampling technique)

10- SMOTEWB (SMOTE with boosting) (Sağlam and Cengiz 2022)

More methods will be implemented.

# R installation

```r
devtools::install_github("fatihsaglam/SMOTEWB")
```

## References

Sağlam, Fatih, and Mehmet Ali Cengiz. 2022. “A Novel SMOTE-Based
Resampling Technique Trough Noise Detection and the Boosting Procedure.”
*Expert Systems with Applications* 200: 117023.
https://doi.org/<https://doi.org/10.1016/j.eswa.2022.117023>.
