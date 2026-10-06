# World-model experiments

225 training days, 180 held-out days, 15 cities.


## E1 one-step NLL (held-out, seen policies)

| model       |    nll |
|:------------|-------:|
| gru_cons    | 0.338  |
| gru         | 0.3412 |
| gru_cons_os | 0.3424 |
| gru_noact   | 0.3435 |
| mlp         | 0.3436 |
| mlp_noact   | 0.3455 |
| gru_os      | 0.3496 |
| glm         | 0.4585 |
| persistence | 0.6781 |
| seasonal    | 0.7709 |


## E2 rollout NLL at 5 / 30 / 60 min, and share of diverging rollouts

| model       |     1 |     6 |     12 |   diverged |
|:------------|------:|------:|-------:|-----------:|
| glm         | 0.462 | 1.011 | 87.685 | 0.512698   |
| gru         | 0.346 | 0.456 |  0.505 | 0.215873   |
| gru_cons    | 0.343 | 0.461 |  0.504 | 0.00857143 |
| gru_cons_os | 0.347 | 0.431 |  0.456 | 0.0203175  |
| gru_os      | 0.354 | 0.487 |  0.625 | 0.273651   |
| mlp         | 0.348 | 0.463 |  0.534 | 0.206667   |
| persistence | 0.688 | 0.935 |  1.006 | 0.0203175  |
| seasonal    | 0.776 | 0.78  |  0.783 | 0.0146032  |


## E3 interactability

```json
{
 "mlp_nll_with_actions": 0.3435797393321991,
 "mlp_nll_without_actions": 0.34547147154808044,
 "mlp_idle_nll_with_vs_without": [
  0.646465539932251,
  0.6494819521903992
 ],
 "gru_nll_with_actions": 0.341154545545578,
 "gru_nll_without_actions": 0.3435327708721161,
 "gru_idle_nll_with_vs_without": [
  0.6449391841888428,
  0.6498242020606995
 ],
 "gru_idle_mae_in_dispatch_target_zones": 0.2934432625770569,
 "gru_noact_idle_mae_in_dispatch_target_zones": 0.5891709327697754,
 "gru_cons_os_nll_seen_policies": 0.3424336612224579,
 "gru_cons_os_nll_unseen_policy_p2_ilp": 0.3487662076950073,
 "mlp_nll_seen_policies": 0.3435797393321991,
 "mlp_nll_unseen_policy_p2_ilp": 0.34952041506767273,
 "policy_gap_corr": 0.8744779023236103,
 "policy_gap_corr_noact_control": 0.7815000453705322,
 "policy_gap_sign_agreement": 0.7227272727272728,
 "policy_gap_n": 9900
}
```


## E4 generalization to unseen cities

| split    | model                                 |    nll |   rollout_nll_1h |   tot_err_new_1h |   tot_err_idle_1h |
|:---------|:--------------------------------------|-------:|-----------------:|-----------------:|------------------:|
| _nosmall | pooled gru_cons_os (saw these cities) | 0.1664 |           0.2373 |           0.6852 |            0.7674 |
| _nosmall | gru_cons_os trained_nosmall           | 0.1696 |           0.2509 |           0.7605 |            0.8706 |
| _nosmall | persistence                           | 0.331  |           0.5015 |           0.8107 |            0.78   |
| _india   | pooled gru_cons_os (saw these cities) | 0.3148 |           0.4088 |           0.4405 |            1.155  |
| _india   | gru_cons_os trained_india             | 0.3211 |           0.4462 |           0.4834 |            3.0444 |
| _india   | persistence                           | 0.6999 |           0.9471 |           0.5125 |            1.0491 |


## E5 plausibility (3-h evening rollouts)

| model       |   impossible_pickup_rate |   fleet_size_rel_err_3h |
|:------------|-------------------------:|------------------------:|
| glm         |                   0.0009 |                215.209  |
| mlp         |                   0.0023 |                  0.3765 |
| gru         |                   0.0034 |                  0.279  |
| gru_os      |                   0.0042 |                  1.1297 |
| gru_cons    |                   0.0058 |                  0.2244 |
| gru_cons_os |                   0.0051 |                  0.1694 |


## E6 fairness observable: zone completion s_z over 3-h rollouts

| model       |   s_mae |   s_corr |   s_std_actual |   s_std_pred |
|:------------|--------:|---------:|---------------:|-------------:|
| glm         |  0.5199 |   0.1052 |         0.1542 |       0.0215 |
| gru         |  0.3367 |   0.167  |         0.1542 |       0.1903 |
| gru_cons    |  0.453  |   0.178  |         0.1542 |       0.1887 |
| gru_cons_os |  0.164  |   0.2063 |         0.1542 |       0.135  |
| gru_os      |  0.1651 |   0.1688 |         0.1542 |       0.13   |
| mlp         |  0.3188 |   0.1659 |         0.1542 |       0.1352 |

