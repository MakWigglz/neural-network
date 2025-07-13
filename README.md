# neural-network
practicing neural networks
Problem: We want to train a neural network to classify whether a student passes or fails an exam based on two factors:

Hours Studied

Previous Exam Scores

Target Output:

0 (Fail)

1 (Pass)

Network Architecture:
Let's build a very simple feedforward neural network.

Input Layer: 2 neurons (for Hours Studied and Previous Scores)

Hidden Layer 1: 3 neurons (this is our first hidden layer)

Output Layer: 1 neuron (for Pass/Fail prediction)

In the context of the definition:

We have L−1=1 hidden layer, so L=2. This means our prediction function will be a composition of L=2 functions: f 
1
​
  (for the hidden layer) and f 
2
​
  (for the output layer).

The overall prediction function will be  
Y
^
 =f 
2
​
 (f 
1
​
 (X)), where X is our input.

Example: A Basic Neural Network for Exam Prediction
1. Input Data (X) and Expected Output (Y)
Let's assume we have the following training data:

Hours Studied (x1)

Previous Score (x2)

Pass/Fail (y)

0.1

0.9

1

0.3

0.2

0

0.8

0.7

1

0.2

0.4

0

0.6

0.5

1

(Note: Inputs are normalized between 0 and 1)

2. Network Parameters (Weights and Biases)
Before training, these are typically initialized randomly. For this example, let's just pick some arbitrary (but illustrative) values.

Weights for Layer 1 (Input to Hidden Layer): W 
1
​
  (shape: 2x3, because 2 inputs, 3 hidden neurons)

W 
1
​
 =( 
0.1
0.2
​
  
0.4
0.5
​
  
0.7
0.8
​
 )
Biases for Layer 1: b 
1
​
  (shape: 1x3)

b 
1
​
 =( 
0.1
​
  
0.2
​
  
0.3
​
 )
Weights for Layer 2 (Hidden Layer to Output Layer): W 
2
​
  (shape: 3x1, because 3 hidden neurons, 1 output neuron)

W 
2
​
 = 

​
  
−0.1
0.3
0.6
​
  

​
 
Biases for Layer 2: b 
2
​
  (shape: 1x1)

b 
2
​
 =( 
−0.2
​
 )
3. Activation Functions
Hidden Layer: We'll use the Sigmoid function for its non-linearity, defined as:

σ(z)= 
1+e 
−z
 
1
​
 
Output Layer: We'll also use Sigmoid since it's a binary classification and outputs a probability between 0 and 1.

The "Composition of Functions" in Action (Forward Pass)
Let's take a single input example: Student 1 (Hours: 0.1, Score: 0.9), expected output Y=1.

Input: X=( 
0.1
0.9
​
 )

Step 1: Layer 1 Transformation (f 
1
​
 ) - Input to Hidden Layer
This is the first function in our composition, f 
1
​
 (X).

Linear Combination (Z 
1
​
 ):

Z 
1
​
 =X 
T
 W 
1
​
 +b 
1
​
 
Z 
1
​
 =( 
0.1
​
  
0.9
​
 )( 
0.1
0.2
​
  
0.4
0.5
​
  
0.7
0.8
​
 )+( 
0.1
​
  
0.2
​
  
0.3
​
 )
Z 
1
​
 =( 
(0.1×0.1+0.9×0.2)
​
  
(0.1×0.4+0.9×0.5)
​
  
(0.1×0.7+0.9×0.8)
​
 )+( 
0.1
​
  
0.2
​
  
0.3
​
 )
Z 
1
​
 =( 
(0.01+0.18)
​
  
(0.04+0.45)
​
  
(0.07+0.72)
​
 )+( 
0.1
​
  
0.2
​
  
0.3
​
 )
Z 
1
​
 =( 
0.19
​
  
0.49
​
  
0.79
​
 )+( 
0.1
​
  
0.2
​
  
0.3
​
 )
Z 
1
​
 =( 
0.29
​
  
0.69
​
  
1.09
​
 )
Activation (A 
1
​
 ): Apply Sigmoid to Z 
1
​
 . This is the output of f 
1
​
 (X).

A 
1
​
 =σ(Z 
1
​
 )=( 
σ(0.29)
​
  
σ(0.69)
​
  
σ(1.09)
​
 )
A 
1
​
 ≈( 
0.572
​
  
0.666
​
  
0.748
​
 )

(These values become the "features" learned by the first layer.)

Step 2: Layer 2 Transformation (f 
2
​
 ) - Hidden to Output Layer
This is the second function in our composition, f 
2
​
 (A 
1
​
 ). The output of f 
1
​
 (X) (which is A 
1
​
 ) becomes the input to f 
2
​
 .

Linear Combination (Z 
2
​
 ):

Z 
2
​
 =A 
1
​
 W 
2
​
 +b 
2
​
 
Z 
2
​
 =( 
0.572
​
  
0.666
​
  
0.748
​
 ) 

​
  
−0.1
0.3
0.6
​
  

​
 +( 
−0.2
​
 )
Z 
2
​
 =(0.572×−0.1)+(0.666×0.3)+(0.748×0.6)+(−0.2)
Z 
2
​
 =−0.0572+0.1998+0.4488−0.2
Z 
2
​
 =0.3914
Activation ( 
Y
^
 ): Apply Sigmoid to Z 
2
​
 . This is the final prediction  
Y
^
 =f 
2
​
 (A 
1
​
 ).

Y
^
 =σ(Z 
2
​
 )=σ(0.3914)
Y
^
 ≈0.596
Interpretation of Result
For Student 1 (who studied 0.1 hours and scored 0.9 previously), our network predicted a probability of 0.596 of passing.
The actual result for Student 1 was 1 (Pass).

Since 0.596 is closer to 1 than to 0, if we use a threshold of 0.5, our network correctly classified this student as passing.

The "Composition" Concept Summarized
The entire process above can be written as:

Y
^
 =σ((σ(X 
T
 W 
1
​
 +b 
1
​
 ))W 
2
​
 +b 
2
​
 )
Let f 
1
​
 (X)=σ(X 
T
 W 
1
​
 +b 
1
​
 ) (This is the function of the first layer, taking input X and producing A 
1
​
 )

Let f 
2
​
 (A 
1
​
 )=σ(A 
1
​
 W 
2
​
 +b 
2
​
 ) (This is the function of the second layer, taking A 
1
​
  as input and producing  
Y
^
 )

Therefore, the final prediction function is indeed the composition:

Y
^
 =f 
2
​
 (f 
1
​
 (X))
This example, though simple, clearly shows how the output of one layer acts as the input to the next, building up complex transformations through the composition of simpler functions. This is the fundamental idea behind deep learning.
