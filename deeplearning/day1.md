6/18/2026

Takes long time to be start the day1, got busy with the new job.

##Day1: Purchase Deep Learning Andrew Ng Coursea, lets start the first day, can't go wrong with Andrew Ng

1.  Neural Network Basics with house price predication as an example. bascially y = k\*x [k is any operation], one neruon one function, easy!!!,
2.  Supervised Machine learning, so it is important to pick waht is x and what is y, got confused about this term:

    Supervised = learn with answers.
    Unsupervised = find patterns without answers.

    Example with animals:

    Supervised:

    Image → Label
    cat image → cat
    dog image → dog

    Unsupervised:

    Many animal images → model groups similar images together
    Group 1: cats
    Group 2: dogs
    Group 3: birds
    - House price prediction Standard NN
    - Image Recognition CNN
    - Audio is played over time, one-dimension tempo sequence RNN
    - Language Chinese-English Translation Complex RNN
    - Automus Driving/Rdar Custom/Hybrid NN

    I don't understand /todo here

    `random thought, How Timbre is digitalized?`

    Pitch = main frequency
    Loudness = amplitude
    Timbre = shape of frequencies + how they change over time

    Only one frequency

    But real instruments contain many frequencies.

    For example, if the note is 440 Hz:

    Fundamental frequency: 440 Hz
    Harmonic 2: 880 Hz
    Harmonic 3: 1320 Hz
    Harmonic 4: 1760 Hz
    ...

    Different instruments have different harmonic strengths.

    Example:

    Piano A4:
    440 Hz strong
    880 Hz medium
    1320 Hz weak
    higher harmonics decay fast

    Violin A4:
    440 Hz strong
    880 Hz strong
    1320 Hz strong
    many higher harmonics

    That different harmonic pattern is a big part of timbre. 3. Computer uses FFT / spectrogram to see timbre

    The raw waveform is hard to understand directly, so we often convert it into frequency information using FFT.

    Waveform → FFT → frequency spectrum

    A spectrum tells us:

    How much energy exists at each frequency

    For changing sound, we use a spectrogram:

    time × frequency × intensity

    So instead of only asking “what frequencies exist,” the computer asks:

    At each moment, what frequencies are strong?

    Sigmoid:
    f(x)=1+e−x1​

    very negative → almost 0
    middle → around 0.5
    very positive → almost 1

    It is good when the output means probability.

    Example:

    Is this email spam?

    model output = 0.91
    meaning: 91% likely spam

    So sigmoid is often used in the last layer for binary classification:

    cat or not cat
    spam or not spam
    disease or no disease

    Problem: sigmoid can become “too flat” near 0 or 1, so learning becomes slow. 2. ReLU

    ReLU means:

    ReLU(x) = max(0, x)

    So:

    ReLU(-5) = 0
    ReLU(-1) = 0
    ReLU(0) = 0
    ReLU(3) = 3
    ReLU(10) = 10

    f(x)=max(0,x)

    Think of ReLU as a gate:

    negative signal → block it, output 0
    positive signal → pass it through

    Example:

    Input z = -2
    ReLU output = 0

    Input z = 5
    ReLU output = 5

    ReLU is commonly used in hidden layers because it is simple, fast, and trains well.

    Main difference
    Function Output range Common use Intuition
    Sigmoid 0 to 1 Final layer for probability “How likely?”
    ReLU 0 to infinity Hidden layers “Pass positive signal, block negative signal”
    Simple rule

    For beginners, remember this:

    Hidden layers → usually ReLU
    Binary output probability → sigmoid

    Example neural network for house price:

    Input features
    → Dense layer + ReLU
    → Dense layer + ReLU
    → Output price

    Example neural network for spam detection:

    Input email
    → Dense layer + ReLU
    → Dense layer + ReLU
    → Sigmoid output

    Output:

    0.95 = spam
    0.10 = not spam

    Very simple mental model:

    ReLU = used inside the brain of the model
    Sigmoid = used at the end when you want probability

    `Random thought, Compute Power is so strong, Mahcine Learning development could have faster iteration, what if we ask LLM to build Machine Learning Code (identify feature, build algorithm, fast iteration), then we can replace LLM with Machine learning on many LLM application using cheaper solution. Would it be possible the world is change from ML=>LLM=>ML`

A new Name: Geoggery Hinton

Backpropagation

Forward pass:
Use weights to make prediction.

Loss:
Measure how wrong the prediction is.

Backprop:
Compute how responsible each weight is for the loss.

Gradient descent:
Change weights to reduce future loss.

An activation function is a small function inside a neural network that decides how much signal a neuron should pass forward.

ReLU is very common:

ReLU(x) = max(0, x)

Sigmoid squeezes any number into between 0 and 1:

sigmoid(x) = 1 / (1 + e^-x)

Imagine a neuron is asking:

Should I activate or not?
How strongly should I pass this information?

Function Output Common use
ReLU 0 to infinity Hidden layers
Sigmoid 0 to 1 Binary probability
Tanh -1 to 1 Older RNNs
Softmax probabilities adding to 1 Multi-class classification

`YOYOYO Softmax it is very fast`

Softmax is an activation function that turns a list of numbers into probabilities.

It is usually used at the last layer of a neural network for multi-class classification.

Example: classify an image as:

cat, dog, bird

The neural network may output raw scores called logits:

cat: 2.0
dog: 1.0
bird: 0.1

These numbers are not probabilities yet. Softmax converts them into probabilities like:

cat: 0.66
dog: 0.24
bird: 0.10

They always add up to 1:

0.66 + 0.24 + 0.10 = 1

So the model predicts:

cat

because cat has the highest probability.

The formula is:

softmax(x_i) = e^(x_i) / sum(e^(all x))

Meaning each score is exponentiated, then divided by the total.

Intuition

Softmax asks:

Among all possible classes, how confident am I in each one?

For example:

Raw scores: [5, 2, 1]
Softmax: [0.94, 0.05, 0.01]

The biggest score becomes the biggest probability.

Difference from sigmoid

Sigmoid is usually for yes/no:

dog or not dog

Softmax is for choosing one among many:

cat vs dog vs bird

So:

Sigmoid → binary probability
Softmax → multi-class probabilities

Very simple version:

Backpropagation = blame assignment for neural network weights.

Man, I totally forgot what I learn in college, GRADIENT

backprop 1986, not that old, huh

Boltzmann machine

Easy!!!
