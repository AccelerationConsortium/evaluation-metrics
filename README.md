# Gaussian Process (GP) model evaluation metrics

## Introduction: GP model assessment in Bayesian Optimization

In iterative product and process development, Bayesian Optimisation (BO) has emerged as a powerful methodology for efficiently navigating complex parameter spaces. BO typically relies on Gaussian Process (GP) models to construct surrogate representations of expensive‑to‑evaluate objective functions. The effectiveness of any BO procedure is therefore tied to the quality of its underlying GP.

This document collects a set of metrics that can be used to evaluate a GP model's performance.  It is almost certainly incomplete, so contributions are very welcome – especially real‑world thresholds and domain‑specific best practices (e.g. "we deploy when $R^2 > 0.7$").

*A few open questions for the community*
- Which metrics do you usually track and why i.e. which actions do you take based on these metrics? 
- Are there any thresholds that you usually apply (e.g. R² > 0.7 is required to be confident about the model) and if so, are they domain specific, what are the respective values and why are they chosen?
- Often, one first starts BO campaigns using space-filling design strategies (e.g. Latin Hypercube Sampling (LHS), Sobol). How do you decide when to transition from a space-filling to a model-driven design approach? Waiting for too long, might waste work and budget on exploration while exploitation would already be more beneficial; doing it too early, however, might result in an inefficient space exploration
- What are your best practices for kernel selection and hyper‑parameter training?

## Create a space-filling design using Latin Hypercube Sampling (LHS)

In this repo, the training and test data are generated using LHS, a statistical method for generating near-random samples with more even coverage than pure random sampling. In our context, we just use it to generate some sample data to calculate the metrices we are interested in.

**Key properties:**
- Ensures samples are distributed evenly across each dimension
- Better space coverage with fewer samples compared to simple random sampling
- Helps prevent clustering of points that can occur with random sampling

LHS can used to generate the initial design points for training the GP model before transitioning to model-driven optimization. There exist several alternative approaches, such as Sobol.

## R² (Coefficient of determination)

R² measures how well the model's predictions match the actual values.

**Mathematical definition:**
```
R² = 1 - (sum of squared residuals) / (total sum of squares)
```

**Interpretation:**
- R² typically ranges from 0 to 1 (in some cases can be negative for poor models)
- Higher R² values indicate a better fit i.e. that the model's predictions are better than of models
with lower R²

**Implementation:**
Here, R² is calculated by comparing model predictions on test data against true function values using sklearn's `r2_score` function.

## Feature importance from length scales

In GPs with Automatic Relevance Determination (ARD) kernels, the learned length scale parameters provide insight into which input dimensions most strongly influence the output.

**Mathematical basis:**
- A length scale (ℓ) is learned for each input dimension
- Smaller length scales indicate greater sensitivity to that dimension
- Importance is calculated as normalized inverse length scales: `imp = (1/ℓ) / sum(1/ℓ)`

**Interpretation:**
- Higher importance values indicate dimensions with stronger influence on the function

This metric can help to determine whether the model has correctly learned the intrinsic dimensionality of the problem.

## Leave-One-Out Pseudo-Likelihood (LOO-PL)

LOO-PL evaluates the predictive accuracy by systematically holding out each training point and predicting it using a model trained on all remaining points. This metric is based on the work of Rasmussen & Williams (2006) and can be efficiently computed using GPyTorch's `LeaveOneOutPseudoLikelihood` function.

**Mathematical Basis:**
For a Gaussian Process model, the leave-one-out predictive log probability for point *i* is (Rasmussen & Williams (2006), eq 5.10):

$$\log p\bigl(y_i \mid X, y_{-i},\theta\bigr) = -\tfrac12\log \sigma_i^{2} -\tfrac{\bigl(y_i - \mu_i\bigr)^2}{2\,\sigma_i^{2}}-\tfrac12\,\log 2\pi,$$

with

$$\mu_i = y_i - \frac{\bigl[K^{-1}y\bigr]_i}{\bigl[K^{-1}\bigr]_{ii}},$$

and

$$\sigma_i^{2} = \frac{1}{\bigl[K^{-1}\bigr]_{ii}}.$$

The total LOO-PL is then the sum over all points (Rasmussen & Williams (2006), eq 5.11):

$$\text{LOO-PL} = \sum_{i=1}^n \log p(y_i | X, y_{-i}, \theta)$$

Note that while Rasmussen & Williams (2006) use the sum, the implementation in this repo implementation takes the mean by dividing by *n*:

$$\text{LOO-PL}_{\text{mean}} = -\frac{1}{n}\sum_{i=1}^n \log p(y_i | X, y_{-i}, \theta)$$

This normalization has several practical advantages:

1. **Interpretability**: The mean represents the average negative log-likelihood per point, making it easier to understand the model's performance on a per-observation basis.

2. **Dataset size independence**: The normalized metric allows for fair comparison between models trained on datasets of different sizes. Without normalization, the sum would grow with dataset size, making it harder to compare models across different scales.

3. **Consistent scale**: The mean version maintains a consistent scale regardless of the number of observations, making it easier to set thresholds or compare against other metrics.

The choice between sum and mean is primarily a matter of practical convenience rather than theoretical difference - both versions contain the same information about model performance, just on different scales.

**Implementation considerations:**
While naive LOOCV would require completely refitting the model *n* times for *n* data points, *GPyTorch*'s implementation optimizes performance by:

1. Reusing the hyperparameters from the original fully-trained model
2. Only recalculating the posterior distribution with the held-out training point

This approach balances computational efficiency with prediction accuracy, avoiding the need to refit hyperparameters for each held-out point. It further penalizes both over-confident and under-confident predictions by considering both the predictive mean and variance.

**Interpretation:**
- Lower values indicate better predictions
- The metric penalizes models that are either too confident (small variance) or too uncertain (large variance) in their predictions

## Kendall's *tau* rank correlation

Kendall's *tau* measures the ordinal association between the model's predictions and true values, focusing on whether the model correctly ranks outputs even if absolute values differ.

**Mathematical Basis:**
```
τ = (number of concordant pairs - number of discordant pairs) / total number of pairs
```

**Interpretation:**
- τ ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation)
- Critical for optimization problems where finding the best points matters more than exact prediction

This metric is particularly valuable when the model will be used for optimization tasks where relative ordering is more important than precise value prediction.

## Conclusion and call for contributions

This document is a living reference and provide only a starting framework for assessing GP model quality in Bayesian Optimization contexts. However, they represent just one perspective in a still evolving field. If you have additional metrics, empirical thresholds or domain‑specific insights, please open an issue or a pull request! The same holds for concerns and comments about the ones already described, of course.

It would be fantastic if experts in the field could contribute their experiences, alternative metrics and case studies to show how they address their GP models' performance, which actions they take based on this performance and also how people decide when to transition from a space-filling to a model-driven approach. 