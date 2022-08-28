# Iterative Loop of ML Development
* Steps of the Loop
    1. Choose the Architecture (model, data, ...)
    2. Train the Model
    3. Diagnose error (bias, variance, error-analysis)
    4. Repeat the cycle until satisfied
* Example: Email Spam classifier:
    1. Architecture: Have a list of the top 10,000 words in the english dictionary. The input $\vec{x}$ will have boolean values stating whether or not the word appeared in the email.
    2. Model: Train logistic regression or Neural Network
    3. Improvements:
        * Collect more data
        * Factor in the email address and subject
        * Make similar words "discount" and "discounting" refer the same thing.
        * Fix misspelled words -> "w4tches" and "med1cine".

# Error Analysis
* If an algorithm has 100 misclassifications out of 500. **Error analysis** is manually looking through and determining patterns. What is common the majority or errors.
    * After determining what traits cause the most errors. Adjust the model to fix these errors.
    * For example: Lets say one of the larger errors is pharma-emails. Then collecting more data or features on phara emails might help.
* Note: Error analysis usually helps for problems that humans are good at.

# Adding Data
* May seem like getting more data is always the solution:
    * This may be expensive and time consuming
    * Instead: Add more data that the error analysis suggests

        For example: Quickly searching through unlabeled data to find th type that I want (phara email).
* Data Augmentation (Really good for images and audio data):
    * Creating more training examples based on the ones we already have.
    * Example: Rotating, enlarging, and recoloring a letter A, could be used a more training data
    * Distortion: See [page 55](Lecture.pdf). Placing a grid on top of an image and running a random walk will create a more robots variety of images.
    * Example 2: Adding a noisy background to a speech recognition audio clip.
    * **NOTE**: You goal is to distort / make more noisy, you data in ways that might actually happen in the test set
* Data Synthesis: (Computer vision tasks)
    * Create more data (out of thin air)
    * Example: Using the computers build in fonts to generate images for OCR training

# Transfer Learning: Using Data from a Different Task
* Transfer Learning: (use when you have limited data)
    * Can only be used with the same data types (images, ...)
* Example: Recognize handwritten digits
1. Use a trained neural network that classifies cats, dogs, people, ... (supervised pre-training)

    This model should be trained from a much larger dataset. Maybe it can be copied from someone who has already spent time training the model.
2. Copy over the trained parameters
3. Train either the output layer or all layers based on the data you have (fine tuning)
* The reason this works (see [page 65](Lecture.pdf)): Every neural network learns to see edges and corners. Using that same learning can be applied to virtually any task.

# Full Cycle of Machine Learning Project
* The full steps for a real ML project
    1. Scope Project: Define what the project is
    2. Collect Data: Define and collect data
    3. Train Model: Training, error analysis ([Bias/Variance](2-BiasAndVariance.md)/[Error Analysis](#error-analysis)), [iterative improvement](#iterative-loop-of-ml-development). Go back to step 2 if necessary
    4. Deploy in Production: Deploy, monitor, and maintain system. Go back to step 2 or 3 if necessary. See the next bullet for info about deployment.
* Deployment:
    * Inference Server: Contains the ML model. When it receives and API call, return the "inference" $\hat{y}$.
    * Mobile App: Interface. Makes and API call to the inference server ($x$).
    * This deployment will require software engineering: (MLOps - Machine learning operations)
        * Reliable / Efficient predictions
        * Scaling
        * Logging
        * System Monitoring:
        * Model Updates

# Fairness, Bias, and Ethics
* ML algorithms might have bias to them: Ex. Hiring tool that discriminates against women
* Adverse use cases: Ex. Spreading toxic speech in social media to optimize for user engagement
* Guidelines:
    * Having a diverse teams that can emphasize potential harm with unique incites
    * Read up on guidelines for the industry
    * Audit system against possible harm before the deployment
    * Develop mitigation plan and after deployment monitor for possible harm

# Quiz: 100%
Quiz [file](Quizzes.md#machine-learning-development-process)