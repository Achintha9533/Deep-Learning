# Infix to Postfix Notation Translation with Neural Networks

The purpose of this project is to implement a neural network that performs the translation of mathematical formulae from traditional **infix notation**—where the operator appears between two operands—to **postfix** (also known as Reverse Polish Notation), where the operator follows the operands.

Infix notation is the most commonly used in human-readable mathematics (e.g., a + b), but it is inherently ambiguous without additional syntactic aids such as parentheses or operator precedence rules. This ambiguity arises because different parse trees can correspond to the same expression depending on how operations are grouped.

In contrast, postfix notation eliminates the need for parentheses entirely. The order of operations is explicitly encoded by the position of the operators relative to the operands, making it more suitable for stack-based evaluation and easier to parse programmatically.

**Example:**

Consider the ambiguous infix expression:
`a + b * c`

This expression can be parsed in at least two different ways:

* **Interpretation (Infix):** `(a + b) * c`
    * **Equivalent Postfix:** `ab+c*`

* **Interpretation (Infix):** `a + (b * c)`
    * **Equivalent Postfix:** `abc*+`

This project takes a data-driven approach to learn the correct postfix form from a given infix expression.

### Constraints

* Any neural network architecture can be used (decoder-only, encoder-decoder, or other).
* The model must have a maximum of 2 million parameters.
* Beam search is not allowed for decoding.
* The core logic of the formula generator, especially the frequency distribution of formulas by depth, should be preserved.
* The model can be trained on a fixed dataset or using an on-the-fly generator.