# Cutting-Plane Active Learning (CPAL)

Cutting-plane active learning is a novel active learning algorithm designed for supervised learning tasks using ReLU networks of arbitrary depths. This active learning scheme induces a gradient-free learning scheme and, to the best of our knowledge, is the first active learning algorithm as of date to have achieved theoretical convergence guarantees. 

🔗 Paper link: [Active Learning of Deep Neural Networks via Gradient-Free Cutting Planes](https://arxiv.org/pdf/2410.02145?)

![CPAL pipeline](documentation/cpal.png)

🚨 ***Note:*** *This repository is under active development. This version is preliminary and subject to change.*

# Abstract
Active learning methods aim to improve sample complexity in machine learning. In this work, we investigate an active learning scheme via a novel gradient-free cutting-plane training method for ReLU networks of arbitrary depth and develop a convergence theory. 
We demonstrate, for the first time, that cutting-plane algorithms, traditionally used in linear models, can be extended to deep neural networks despite their nonconvexity and nonlinear decision boundaries. Moreover, this training method induces the first deep active learning scheme known to achieve convergence guarantees, revealing a geometric contraction rate of the feasible set. We exemplify the effectiveness of our proposed active learning method against popular deep active learning baselines via both synthetic data experiments and sentimental classification task on real datasets.

# Repo Structure



# Citation

If you find our work useful, consider giving us a star ⭐ and citing our paper as:

<pre> \ @inproceedings{
    zhang2025active,
    title={Active Learning of Deep Neural Networks via Gradient-Free Cutting Planes},
    author={Erica Zhang and Fangzhao Zhang and Mert Pilanci},
    booktitle={The Forty-Second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=SmYDdeLAR5}
}</pre>
