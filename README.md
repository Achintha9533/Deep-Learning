### Project Description

This project presents a neural network-based solution for the automated translation of mathematical expressions from traditional **infix notation** to **postfix notation** (Reverse Polish Notation).

Infix notation, where operators are placed between their operands (e.g., $a + b$), is the standard for human-readable mathematics. However, it can be inherently ambiguous without additional syntactic rules or parentheses. In contrast, postfix notation eliminates this ambiguity by explicitly encoding the order of operations through the placement of operators after their operands, making it ideal for stack-based evaluation and programmatic parsing.

**Example:**

Consider the ambiguous infix expression: $a + b * c$

This expression can be parsed in two distinct ways, each corresponding to a different postfix form:

* **Interpretation:** $(a + b) * c$
    * **Equivalent Postfix:** $ab+c*$
* **Interpretation:** $a + (b * c)$
    * **Equivalent Postfix:** $abc*+$

The core objective of this project is to train a model that learns the rules of operator precedence and associativity to correctly disambiguate infix expressions and generate their corresponding postfix forms. To manage the complexity of this learning task, the dataset is restricted to expressions with a maximum syntactic depth of 3, ensuring a bounded and manageable set of structural possibilities.

### Technical Constraints

The neural network implementation must adhere to the following technical constraints:

* **Architecture:** Any neural network architecture, including decoder-only, encoder-decoder, or other variants, is permitted.
* **Parameter Limit:** The total number of trainable parameters must not exceed 2 million.
* **Decoding Strategy:** The decoding process must use a greedy search algorithm; beam search is not allowed.
* **Data Generator:** The core logic of the formula generator, particularly the frequency distribution of expressions by depth, must be preserved. Adaptations for efficiency are permitted.
* **Dataset:** The model may be trained using a pre-generated fixed dataset or an on-the-fly generator.

### Evaluation

Model performance will be evaluated using a custom metric called **prefix accuracy**. For a generated output $y_{pred}$ and a ground truth $y_{true}$, the score is defined as the length of the longest matching initial prefix, divided by the maximum length of the two sequences (up to the End-of-Sequence token). A perfect match yields a score of 1.0.

This metric is highly informative because:
* It provides a continuous score where an exact match score would often be 0.
* It is a more precise measure of sequential generation quality than edit distance.
* It effectively pinpoints where the model's generation process first deviates from the ground truth.

### Deliverables

The final submission must consist of the following:

* A single Python notebook written in Keras.
* The notebook must be a standalone file; a ZIP archive will be rejected.
* The notebook should contain clear documentation of the training phase and its history.
* A mechanism to provide the network parameters is required. A suggested method is to include a `gdown` script within the notebook to upload or download model weights.