# Understanding Cross Entropy

**Bhaskar S** | *07/12/2025*

---

For a **Classification** model that predicts one of the $M$ classes (or target labels), given an input, the model predicts the probability distribution for the $M$ classes.

For the classification model training, we need a loss function that compares two probability distributions - one the ground truth (real) probability distribution and the other the predicted probability distribution, and penalize the probabilities from the predicted distribution that are confidently wrong.

This is where the **Cross Entropy** loss function comes into play!!!

The term **Entropy** is for measuring the uncertainity in the prediction and term **Cross** is for comparing two distributions - the ground truth vs the prediction.

In mathematical terms, the Cross Entropy loss function is expressed as follows:

$H(P, Q_{\theta}) = - \sum_i^N P(y_i|x_i) . \log_e(Q_{\theta}(\hat{y_i}|x_i))$ $(1)$

where

$P(y_i|x_i)$ is the ground truth probability distribution with real probabilities $y_i$ given the input $x_i$ (represented as $y_i|x_i$)

and

$Q_{\theta}(\hat{y_i}|x_i)$ is the predicted probability distribution with probabilities $\hat{y_i}$ given the input $x_i$ (represented as $\hat{y_i}|x_i$) with $\theta$ being the model parameters

## Unpacking Cross Entropy

Let us try to unpack the Cross Entropy loss equation shown in the equation $(1)$.

One way to compare two quantities is using **Subtraction**. if the result is zero, then the two quantities are equal else not.

But, how does one perform a subtraction on two probability distributions???

One approach would be to find the **Mean** (or **average**) for each of the probability distributions.

The next challange - how does one find the mean of a probability distribution???

This is where we can leverage the **Expected Value**, which approximate to the average for very large sample sizes (law of large numbers).

Mathematically, the Expected Value can be expressed as follows:

$\mathbb{E}(X) = \sum_i^N x_i . p(x_i)$ $(2)$

where

$X$ is the random variable representing the sample space and $x_i$ is the $i^{th}$ sample from the sample space,

and

$p(x_i)$ is the probability for the sample $x_i$

## Measuring Surprise

We indicated the intent of the Cross Entropy is to **penalize** the model for assigning higher probabilities to incorrect target classes. In other words, Cross Entropy is about the measure of **surprise**. For example, when an incoming email is classified as SPAM when it is indeed SPAM (high predicted probability), we are less surprised (or low Entropy value). However, if the incoming email is NOT classified as SPAM when it is indeed a SPAM (low predicted probability), we are more surprised (or high Entropy value). This implies there is an inverse relation between the predicted probability and the Entropy. This is where one can leverage a **Logarithm** scale for Entropy (the inverse behavior).

For example, if the predicted probability is $0.8$, then $log_e(0.8) \approx -0.22$.

Similarly, if the predicted probability is $0.1$, then $log_e(0.1) \approx -2.30$.

One can compute the Expected Value (equation $(2)$ from above) for probability distributions as well.

Mathematically, the Expected Value for a probability distribution can be expressed as follows:

$\mathbb{E}(p(X)) = - \sum_i^M p(x_i) . log_e(p(x_i))$ $(3)$

where

$p(x_i)$ is the probability for the sample $x_i$

and

$log_e(p(x_i))$ is a way to measure the element of surprise

Notice the use of the **minus** sign, because $log_e()$ will return a **negative** value for probability values between $0.0$ and $1.0$, and hence need to compensate for it.

## Testing with an Example

Let us test the above equation $(3)$ with an example to see if it works.

Assume a classification model that predicts one of the $3$ target labels. Let the ground truth be the probabilities $\{0.98, 0.01, 0.01\}$.

If the predicted probabilities are $\{0.8, 0.1, 0.1\}$, then the Expected Value for the predicted distribution will be $-(0.98 * log_e(0.8) + 0.01 * log_e(0.1) + 0.01 * log_e(0.1)) \approx 0.26$.

However, if the predicted probabilities are $\{0.1, 0.8, 0.1\}$, then the Expected Value for the predicted distribution will be $-(0.98 * log_e(0.1) + 0.01 * log_e(0.1) + 0.01 * log_e(0.1)) \approx 4.56$.

Notice the value $4.56$ is much higher (penalizing for wrong prediction) than the value $0.26$.

## Comparing Probability Distributions

Given that we have all the ingredients, we can compare two probability distributions $P$ and $Q_{\theta}$ by using equation $(3)$ and the subtraction operation as follows:

$\sum_i^N P(y_i|x_i) . log_e(P(y_i|x_i)) - \sum_i^N P(y_i|x_i) . log_e(Q_{\theta}(\hat{y_i}|x_i))$ $(4)$

Simplifying the equation $(4)$ from above, we get:

$\sum_i^N P(y_i|x_i) . [log_e(P(y_i|x_i)) - log_e(Q_{\theta}(\hat{y_i}|x_i))]$

Or:

$\sum_i^N P(y_i|x_i) . log_e(\frac{P(y_i|x_i)}{Q_{\theta}(\hat{y_i}|x_i)})$ $(5)$

The equation $(5)$ above is referred to as the **Kullback-Leibler Divergence** or **KL Divergence** for short, and is often used for comparing two probability distributions.

When we train a classification model, we use **Gradient Descent** (derivatives from calculus) during **Backpropagation** to optimize the model weights $\theta$.

Given the first term in the equation $(4)$ above is independent of $\theta$, it can be dropped for parameter optimization.

Hence, we end up with the simplified equation:

$- \sum_i^N P(y_i|x_i) . log_e(Q_{\theta}(\hat{y_i}|x_i))$ $(6)$

Compare the above equation $(6)$ to the equation of Cross Entropy in $(1)$!!!

## Conclusion

Hope this article provides the intuition into why the Cross Entropy loss function for classification models!!!

## References

[**Cross Entropy - PyTorch**](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

---

© PolarSPARC
