# Error metrics for skewed datasets
* For dataset when the number of positive results is much less than the negative results, a typical error analysis won't work.
* Problem: Determining whether or not a patient have a rare disease.
* Instead off calculating the % of time when the algorithms is correct or wrong, create a table:

    Predicted class by Actual Class:
     - | 1 | 0
    ---: | --- | ---
    **1** | 15 | 5
    **0** | 10 | 70
* We can see what percentages of cases happen in each of the four scenarios
    * True positive: (1, 1) - 15
    * False positive: (1, 0) - 5
    * False negative: (0, 1) - 10
    * True negative: (0, 0) - 70
* Other important terms:
    * **Precision** (What percent of patients really have the disease if we predict they do): $\frac{\text{True Pos}}{\text{True Pos + False Pos}} = \frac{15}{15 + 5} = 75\%$
    * **Recall** (What fraction did we correctly guess had the disease of those that had it): $\frac{\text{True Pos}}{\text{True Pos + False Neg}} = \frac{15}{15 + 10} = 60\%$
* Looking at both Precision and Recall determines whether or not a learning algorithm is good enough.

# Trading off precision and recall
* We can increase or decrease the threshold that determines $\hat{y}$.
    * For example be can predict $1$ if $f_{\vec{w},b}(\vec{x}) \ge 0.7$ instead of $0.5$
    * Increasing the threshold will increase the precision: When $\hat{y}=1$, the answer is more likely to be 1.
    * Increasing the threshold will decrease the recall: I am less likely to predict $\hat{y}=1$, when the answer is 1.
* Depending on the circumstance, we can choose the threshold:
    * Ex. If we are predicting a rare disease. It is better to stay on the safe side and set the threshold low.
* We can plot the precision and recall and a graph and pick whatever point on the curve that best matches out needs. (See [page 78](Lecture.pdf)).
* **F1 Score**: Automatically combine precision and recall
    * This is a score that factors in using both precision and recall to create one metric to use
    * One option: Taking the average of precision and recall is not optimal
    * $F_1$ Score / Harmonic Mean: Gives more importance to the lower value - $\frac{1}{\frac12 \left( \frac1P + \frac1R \right)} = 2\frac{PR}{P + R}$