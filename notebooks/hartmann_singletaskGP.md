---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from gpcheck.data import create_hartmann_dataset
from gpcheck.metrics import calculate_metrics
from gpcheck.models import GPConfig, GPModel
from gpcheck.visualization import create_evaluation_plot
```

```python
config = GPConfig()

dim = 6
n_train = 80
n_test = 20

train_X, train_Y, test_X, test_Y, f_true = create_hartmann_dataset(
    seed=42, dim=dim, n_train=n_train, n_test=n_test
)

print(
    f"Created dataset with {dim} dimensions, "
    f"{train_X.shape[0]} training points and "
    f"{test_X.shape[0]} test points"
)
```

```python
gp_model = GPModel(config)
gp_model.fit(train_X, train_Y)
```

```python
metrics = calculate_metrics(gp_model, train_X, train_Y, test_X, test_Y)
```

```python
fig = create_evaluation_plot(metrics)
fig.show()
```

```python
print("\n=== Summary ===")
print(f"r2: {metrics['r2']}")
print(f"imp_cumsum: {metrics['imp_cumsum']}")
print(f"loo_nll: {metrics['loo_nll']}")
print(f"rank_tau: {metrics['rank_tau']}")
```

```python

```
