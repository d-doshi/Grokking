# Grokking

Grokking in machine learning refers to the phenomenon of delayed and sudden generalization. The network quickly reaches 100% training accuracy without generalizing. Upon training longer, the network eventually learns the relevant features and obtains 100% test accuracy. Modular Arithmetic is the canonical task to demonstrate grokking. Although originally it was observed in transformers [1], it is possible to reproduce this phenomenon in fully connected networks (aka Multilayer Perceptrons or MLPs) [2].

## Interpretability

In MLPs, the generalizing features are completely interpretable: the network weights take on a periodic form. In fact, it is possible to write analytical solution for the network weights that gives 100% accuracy [2]. Notably, the grokked networks learn similar features to the analytical solution. The similarity can be studied using measures of periodicity (localization in Fourier space) such as "Inverse Participation Ratio". Such metrics allow us to clearly discern generalization from memorization.

## Label Corruption

Generalization and memorization can be further distinguished by corrupting a fraction of the labels in the training dataset. Deep networks are generally capaable of simultaneously memorize corrupted data *and* generalize on test data. Notably, in this setup, the generalizing and memorizing neurons can be identified and separated. Pruning the memorizing neurons eliminates memorization of corrupted data and gives perfect performance on uncorrupted data. Various forms of regularization such as weight decay, Dropout, BatchNorm prevent memorization of corrupted data and enhance performance on uncorrupted data during optimization. Our setup also makes transparent the effects of these regularization techniques. Weight decay and Dropout force all neurons to learn generalizing (periodic) features, while BatchNorm de-amplifies memorizing neurons and amplifies generalizing ones. For an in-depth discussion, see [3].

# Map

**Notebooks**: Demo notebooks
- Grokking_modular_arithmetic.ipynb : Reproduces grokking on modular addition with a 2-layer MLP [2]. Demonstrates periodic weights, generalizing features etc.
- Grokking_with_label_corruption.ipynb : Reproduces all the results with label corruption [3], including training curves, effect of various regularization techniques, IPR analysis etc.

**Training_scripts**: Training scripts for reproducing the phase diagrams in [3].

**utils**: Utility files with models, datasets, optimization and other helper functions/classes.

# Papers

Currently, this repository reproduces the results from the following papers:
<details>
<summary>
To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic datasets
</summary>

```
@misc{
}
```
</details>
<details>
<summary>
Grokking modular arithmetic (<a href="https://arxiv.org/abs/2301.02679">arXiv</a>) [<b>bib</b>]
</summary>
  
```
@misc{gromov2023grokking,
      title={Grokking modular arithmetic}, 
      author={Andrey Gromov},
      year={2023},
      eprint={2301.02679},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
</details>

# References

[1] Power et. al. "Grokking: Generalization Beyond Overfitting on Small Algorithic Datsets". arXiv:220102177

[2] Andrey Gromov. "Grokking modular arithmetic". arXiv:2301.02679

[3] Doshi et. al. "To grok or not to grok: Distentangling generalization and memorization on corrupted algorithmic datasets." arXiv:
