# Cutting-Plane Active Learning (CPAL)

Cutting-plane active learning is a novel active learning algorithm designed for supervised learning tasks using ReLU networks of arbitrary depths. This active learning scheme induces a gradient-free learning scheme and, to the best of our knowledge, is the first active learning algorithm as of date to have achieved theoretical convergence guarantees. 

🔗 Paper link: [Active Learning of Deep Neural Networks via Gradient-Free Cutting Planes](https://arxiv.org/pdf/2410.02145?)

![CPAL pipeline](documentation/cpal.png)

🚨 ***Note:*** *This repository is under active development. This version is preliminary and subject to change.*

# Abstract
Active learning methods aim to improve sample complexity in machine learning. In this work, we investigate an active learning scheme via a novel gradient-free cutting-plane training method for ReLU networks of arbitrary depth and develop a convergence theory. 
We demonstrate, for the first time, that cutting-plane algorithms, traditionally used in linear models, can be extended to deep neural networks despite their nonconvexity and nonlinear decision boundaries. Moreover, this training method induces the first deep active learning scheme known to achieve convergence guarantees, revealing a geometric contraction rate of the feasible set. We exemplify the effectiveness of our proposed active learning method against popular deep active learning baselines via both synthetic data experiments and sentimental classification task on real datasets.

# Repo Structure

- **`scripts`**: `python` scripts for running various deep active learning methods for synthetic classification or regression tasks. 
    
    The primary scripts are as follows:
    - **`run_deepal_class.py`**: Run various deep active learning algorithms from [DeepAL](https://github.com/ej0cl6/deep-active-learning) on classification of a synthetic spiral. A sample code snipet to run the script is given as follows: <pre>python -m deepal_baseline.demo \ --n_round 3 \ --n_query 10 \ --n_init_labeled 10 \ --dataset_name Spiral \ --strategy_name EntropySampling \ --seed 1
    
    This runs the "Entropy Sampling" deep AL baseline on the synthetic Spiral dataset with 10 initial labels, a total of 3 rounds of querying with 10 queries in each round. 
    - **COMING UP**

- **`deepal_baseline`**: contains adapted codes from [DeepAL](https://github.com/ej0cl6/deep-active-learning) to include regression as well as customization to our methods and datasets.

- **`examples`**: contains `jupyter` notebook tutorials. In paritcular:
    - **`examples/cpal_classification.ipynb`** and **`examples/cpal_regression.ipynb`** gives examples of running CPAL as well as various deep AL baselines used in our paper on both classification and regression tasks. Here, we use the synthetic binary spiral and quadratic regression as an example.
    - **`examples/test_cvxnn_solver.ipynb'** gives examples running the convex solver on the exact equivalent reformulation of a two-layer ReLU Network. It compares this method with the classical gradient descent algorithm using back propogation.
    - **`examples/cpal_3layer.ipynb`** contains examples running deep AL via a three-layer ReLU network. It demonstrates the extendability of our methods to arbitrarily deep NN, given computes, by running the three-layer CPAL on the binary synthetic spiral classification task.
- **`src`**: contains `python` source codes for running CPAL. We highlight a few items:
    - **`src/apps/sentiment_classification.py`** includes the `python` script for running CPAL on a sample sentiment classification task using LLMs.
    - **`src/baselines/skactive_baseline.py`** and **`src/baselines/linear_cpal.py`** contain deep AL baselines from [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml) and the linear cutting-plane method which we adapted from [Louche & Ralaivola (2015)](https://arxiv.org/abs/1508.02986).
    - **`src/cpal/cpal.py`**: contains codes for our core algorithm CPAL on both classification and regression task. In addition, **`src/cpal/cpal_3layer.py`** contains source codes for running CPAL with three-layer ReLU networks. It currently only supports classification tasks, but we are working on extending this functionality to regression.


# Citation

If you find our work useful, consider giving our repo a ⭐ and citing our paper as:

<pre>@inproceedings{
    zhang2025active,
    title={Active Learning of Deep Neural Networks via Gradient-Free Cutting Planes},
    author={Erica Zhang and Fangzhao Zhang and Mert Pilanci},
    booktitle={The Forty-Second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=SmYDdeLAR5}
}</pre>
