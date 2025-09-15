# Infix to Postfix Notation Translation with Neural Networks

This project implements a neural network to translate mathematical formulas from **infix notation** to **postfix notation** (also known as Reverse Polish Notation).

Infix notation is common in human-readable mathematics (e.g., `a + b`), but it can be ambiguous without parentheses or operator precedence rules. Postfix notation, on the other hand, eliminates the need for parentheses because the order of operations is explicitly defined by the placement of operators after their operands, which makes it suitable for stack-based evaluation.

This project uses a data-driven approach with a neural network to learn to generate the correct postfix form from a given infix expression. The dataset is restricted to formulas with a maximum syntactic depth of 3 to simplify the task and manage expression complexity.

### Key Components

* **Expression Generation**: The `generate_infix_expression` function generates mathematical formulas using 5 identifiers (`a`, `b`, `c`, `d`, `e`) and 4 binary operators (`+`, `-`, `*`, `/`).
* **Notation Conversion**: The `infix_to_postfix` function converts a list of infix tokens to postfix notation.
* **Encoding/Decoding**: Functions `encode` and `decode_sequence` are used to convert token lists to integer IDs and back to readable strings.
* **Sequence-to-Sequence Model**: The project uses a sequence-to-sequence (seq2seq) model with a Luong-style dot-product attention mechanism for the translation. The model includes an encoder, a decoder, and an attention layer to focus on relevant input parts during output generation.

---

### Constraints

* Any neural network architecture can be used (decoder-only, encoder-decoder, or other).
* The model must have a maximum of 2 million parameters.
* Beam search is not allowed for decoding.
* The core logic of the formula generator, especially the frequency distribution of formulas by depth, should be preserved.
* The model can be trained on a fixed dataset or using an on-the-fly generator.