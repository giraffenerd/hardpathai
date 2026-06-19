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

Very simple version:

Backpropagation = blame assignment for neural network weights.

Man, I totally forgot what I learn in college, GRADIENT

Easy!!!
