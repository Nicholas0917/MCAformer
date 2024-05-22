# MCAformer
**M**ulti-scale **C**hronological **A**ware Trans**former** (MCAformer) for Long-term Time Series Forecasting

Time series data exhibits inherent temporal dependencies, where the relationships between time steps should be unidirectional and decrease as the distance between the time steps increases. However, commonly used models like Transformers and MLPs often struggle to effectively capture this "chronological order" present in time series data.

To address this challenge, this paper proposes a novel model called the Multi-scale Chronological Aware Transformer (MCAformer). The key innovations of this model are:

1. It incorporates a unidirectional time decay matrix into the traditional attention mechanism to explicitly introduce the temporal ordering of the time series.

![CAAttn](https://github.com/Nicholas0917/MCAformer/assets/49270065/1ea0a5b2-33c8-4db6-a09d-eb468606e6a1)
3. It leverages a multi-head attention approach, where each head uses a different time decay coefficient to construct the unidirectional time decay matrix. This allows the model to capture time series patterns at different time scales.
4. The model also employs a simple gated linear unit to adaptively adjust the weights of different variables, effectively modeling the correlations between them.
5. 
![model_structure](https://github.com/Nicholas0917/MCAformer/assets/49270065/01c7e7b1-8677-4776-9760-2199b441527e)


Experiments on public datasets demonstrate that the proposed MCAformer achieves strong performance, outperforming conventional time series models. 

![result](https://github.com/Nicholas0917/MCAformer/assets/49270065/6ea2097f-252b-43d2-ac22-7888e270150d)
