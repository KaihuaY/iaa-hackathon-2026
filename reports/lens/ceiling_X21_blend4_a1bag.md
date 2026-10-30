# L0 headroom ceiling: X21_blend4_a1bag

```
     line variant  n_rows  n_claims  d2_act  d2_sim_mean  d2_sim_sd         z  excess_share  sub25_sd
    motor  global  725270     84631 0.09234      0.08761    0.00588   0.05980       0.00147   0.00447
    motor  binned  725270     84631 0.09234      0.08779    0.00220   0.08341       0.00171   0.00447
    motor   gamma  725270     84631 0.09234      0.06731    0.00701 -10.48803      -0.26148   0.00447
buildings  global  523725     19695 0.10423      0.09535    0.00314  -0.71263      -0.00524   0.00651
buildings  binned  523725     19695 0.10423      0.09553    0.00403  -0.26438      -0.00252   0.00651
buildings   gamma  523725     19695 0.10423      0.09436    0.00386  -3.29219      -0.03837   0.00651
 contents  global  534126     17682 0.10051      0.09195    0.00392  -1.00548      -0.00938   0.00757
 contents  binned  534126     17682 0.10051      0.09065    0.00362  -0.84421      -0.00594   0.00757
 contents   gamma  534126     17682 0.10051      0.08994    0.00370  -3.14106      -0.02313   0.00757
  overall  global 1783121    122008 0.09783      0.09100    0.00277  -0.23729      -0.00273   0.00364
  overall  binned 1783121    122008 0.09783      0.09093    0.00197  -0.10626      -0.00109   0.00364
  overall   gamma 1783121    122008 0.09783      0.07923    0.00413 -11.58642      -0.14343   0.00364
```

Reading rule: excess_share = (D_act - mean D_sim) / D_null; positive = actual fits worse than if mu were the true mean = possible headroom in D2 units. If |excess_share| < 2*d2_sim_sd the predictions are consistent with truth and no headroom is detectable. sub25_sd is the sd of d2 over 25% row subsets (public-board noise proxy). Global variant: motor: consistent with truth, no detectable headroom; buildings: consistent with truth, no detectable headroom; contents: actual fits better than simulated truth (severity pool too noisy or mu too tight); overall: consistent with truth, no detectable headroom.