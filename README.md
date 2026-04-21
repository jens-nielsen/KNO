## Kernel Neural Operators (KNOs) for Scalable, Memory-efficient, Geometrically-flexible Operator Learning
This paper introduces the Kernel Neural Operator (KNO), a provably convergent operator-learning architecture that utilizes compositions of deep kernel-based integral operators for function-space approximation of operators (maps from functions to functions). The KNO decouples the choice of kernel from the numerical integration scheme (quadrature), thereby naturally allowing for operator learning with explicitly-chosen trainable kernels on irregular geometries. On irregular domains, this allows the KNO to utilize domain-specific quadrature rules. To help ameliorate the curse of dimensionality, we also leverage an efficient dimension-wise factorization algorithm on regular domains. More importantly, the ability to explicitly specify kernels also allows the use of highly expressive, non-stationary, neural anisotropic kernels whose parameters are computed by training neural networks. Numerical results demonstrate that on existing benchmarks the training and test accuracy of KNOs is comparable to or higher than popular operator learning techniques while typically using an order of magnitude fewer trainable parameters, with the more expressive kernels proving important to attaining high accuracy. KNOs thus facilitate low-memory, geometrically-flexible, deep operator learning, while retaining the implementation simplicity and transparency of traditional kernel methods from both scientific computing and machine learning.

```
@misc{lowery2025kernelneuraloperatorsknos,
      title={Kernel Neural Operators (KNOs) for Scalable, Memory-efficient, Geometrically-flexible Operator Learning}, 
      author={Matthew Lowery and John Turnage and Zachary Morrow and John D. Jakeman and Akil Narayan and Shandian Zhe and Varun Shankar},
      year={2025},
      eprint={2407.00809},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.00809}, 
}
```
### Code instructions
Each of the problems have their own respective python script, which can be called directly. The defaults hyperparameters are set, but can be changed via the argparse syntax:


```
python3 burgers.py --int-kernel=ns_gsm --lift-dim=128
```

Otherwise install python 3.10 and then run

```
pip install -U jax[cuda12]
pip install -U equinox
pip install -U optax
```

(changing cuda12 to cpu if necessary)
