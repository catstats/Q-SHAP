# Q-SHAP Algorithm Notes

## General Trees

The stable generic backend remains `T2()` / `T2_sample()`. It uses the existing Q-SHAP complex-root polynomial machinery:

- `store_complex_root()`
- `store_complex_v_invc()`
- `complex_dot_v2()`
- `store_z`
- `store_v_invc`

The experimental `t2_ol2d()` interface is present so the depth-linear unordered leaf-pair kernel can be validated without replacing the stable backend. In this build, `t2_ol2d()` delegates to stable `T2()`.

## CatBoost Trees

CatBoost JSON is normalized into the shared `simple_tree` representation.
`oblivious_trees` retain the optimized symmetric implementation below, while
the nested `trees` emitted by `grow_policy="Depthwise"` and
`grow_policy="Lossguide"` use the stable general-tree backend. Both parsers
map CatBoost float-feature indices back to flat input columns, quantize input
features to float32 before routing, and preserve the per-feature missing-value
direction. Only numeric `FloatFeature` splits are supported.

### Symmetric fast path

The fast CatBoost global backend is separate from the generic `t2_ol2d()` interface.
It groups by parsed leaf/path, not raw prediction value.

For one tree, reached leaf/path, and feature `j`, the backend caches:

- `T0_leaf[j]`
- `T2_leaf[j]`

For samples `G` reaching the same leaf/path:

- `m = |G|`
- `sum_r = sum residual_i`
- `sum_r2 = sum residual_i^2`

With CatBoost JSON leaf values already scaled by learning rate:

```text
a = T2_leaf[j]
b = 2 * T0_leaf[j]

loss_sum[j]   += m*a - b*sum_r
loss_sumsq[j] += m*a*a - 2*a*b*sum_r + b*b*sum_r2
```

The global coefficient of determination is:

```text
R_j^2 = -loss_sum[j] / SST
```

For depth `D`, leaves `L = 2^D`, trees `T`, and reached leaves `U`, the batch complexity is:

```text
O(T [nD + ULD])
```

Worst case:

```text
O(T [nD + L^2D])
```
