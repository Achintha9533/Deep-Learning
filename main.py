# ==============================================================================
# Infix to Postfix Notation Translation with Neural Networks
# ==============================================================================

# Project Description:
# The purpose of this project is to implement a neural network that performs the
# translation of mathematical formulae from traditional **infix notation**—where
# the operator appears between two operands—to **postfix** (also known as Reverse
# Polish Notation), where the operator follows the operands.

# Infix notation is the most commonly used in human-readable mathematics (e.g.,
# a + b), but it is inherently ambiguous without additional syntactic aids such as
# parentheses or operator precedence rules. This ambiguity arises because different
# parse trees can correspond to the same expression depending on how operations
# are grouped.

# In contrast, postfix notation eliminates the need for parentheses entirely. The
# order of operations is explicitly encoded by the position of the operators
# relative to the operands, making it more suitable for stack-based evaluation
# and easier to parse programmatically.

# Example:
# Consider the ambiguous infix expression:
# a + b * c
#
# This expression can be parsed in at least two different ways:
#
# Interpretation (Infix): (a + b) * c
# Equivalent Postfix: ab+c*
#
# Interpretation (Infix): a + (b * c)
# Equivalent Postfix: abc*+
#
# This project takes a data-driven approach to learn the correct postfix form
# from a given infix expression.

# ==============================================================================
# Code
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import random
import gdown

# Constants and Configuration
MAX_DEPTH = 3
NUM_IDENTIFIERS = 5
NUM_OPERATORS = 4
MAX_LEN = 15
VOCAB_SIZE = NUM_IDENTIFIERS + NUM_OPERATORS + 4  # Includes '(', ')', and padding tokens
EMBED_DIM = 64
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 25
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 500
VALIDATION_STEPS = 50

# Global operator precedence and associativity
OP_PRECEDENCE = {'*': 2, '/': 2, '+': 1, '-': 1}
OP_ASSOCIATIVITY = {'*': 'left', '/': 'left', '+': 'left', '-': 'left'}

# Tokens
IDENTIFIERS = [chr(ord('a') + i) for i in range(NUM_IDENTIFIERS)]
OPERATORS = ['+', '-', '*', '/']
TOKENS = IDENTIFIERS + OPERATORS + ['(', ')', '[PAD]', '[UNK]']

# Tokenizers
in_tokenizer = {token: i for i, token in enumerate(TOKENS)}
out_tokenizer = {i: token for i, token in enumerate(TOKENS)}

# Helper functions
def is_identifier(token):
    return token in IDENTIFIERS

def is_operator(token):
    return token in OPERATORS

def precedence(op):
    return OP_PRECEDENCE.get(op, 0)

def generate_expression(depth=0):
    if depth >= MAX_DEPTH or random.random() < 0.5:
        return random.choice(IDENTIFIERS)
    
    op = random.choice(OPERATORS)
    
    left = generate_expression(depth + 1)
    right = generate_expression(depth + 1)

    if random.random() < 0.2:
        return f"({left}{op}{right})"
    return f"{left}{op}{right}"

def infix_to_postfix(infix):
    stack = []
    output = []
    
    infix_list = [char for char in infix if char.strip()]

    for token in infix_list:
        if is_identifier(token):
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif is_operator(token):
            while (stack and precedence(stack[-1]) >= precedence(token) and
                   OP_ASSOCIATIVITY.get(token, 'left') == 'left'):
                output.append(stack.pop())
            stack.append(token)
            
    while stack:
        output.append(stack.pop())
        
    return output

def preprocess_sequence(seq, tokenizer):
    encoded = [tokenizer.get(token, tokenizer['[UNK]']) for token in seq]
    encoded = encoded + [tokenizer['[PAD]']] * (MAX_LEN - len(encoded))
    return np.array(encoded, dtype=np.int32)

# Data Generator
def dataset_generator():
    while True:
        infix_str = generate_expression()
        postfix_list = infix_to_postfix(infix_str)
        
        infix_tokens = [char for char in infix_str if char.strip()]
        
        if len(infix_tokens) > MAX_LEN or len(postfix_list) > MAX_LEN:
            continue
            
        x = preprocess_sequence(infix_tokens, in_tokenizer)
        y = preprocess_sequence(postfix_list, in_tokenizer)
        
        yield x, y

# Build the dataset
dataset = tf.data.Dataset.from_generator(
    dataset_generator,
    output_signature=(
        tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
        tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Encoder
encoder_input = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
encoder_embedding = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(encoder_input)
encoder_lstm = layers.LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_input = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
decoder_embedding = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(decoder_input)
decoder_lstm = layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention (Luong-style)
attention = layers.Attention()
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_concat = layers.Concatenate(axis=-1)([decoder_outputs, context_vector])

# Dense output layer
decoder_dense = layers.Dense(VOCAB_SIZE, activation='softmax')
output = decoder_dense(decoder_concat)

# Model
model = keras.Model([encoder_input, decoder_input], output)

# Training configuration
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Dummy data for validation
dummy_data = next(iter(dataset))
dummy_x = dummy_data[0]
dummy_y = dummy_data[1]

# Training
model.fit(
    dataset.take(STEPS_PER_EPOCH),
    epochs=EPOCHS,
    validation_data=dataset.take(VALIDATION_STEPS)
)

# Inference models
encoder_model = keras.Model(encoder_input, encoder_states)

decoder_state_input_h = layers.Input(shape=(LATENT_DIM,))
decoder_state_input_c = layers.Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]

decoder_outputs_concat = layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
decoder_outputs_final = decoder_dense(decoder_outputs_concat)

decoder_model = keras.Model(
    [decoder_input] + decoder_states_inputs,
    [decoder_outputs_final] + decoder_states
)

# Download and load weights
print("Downloading and loading model weights...")
file_id = "1OwZQ7maaDxedvDgTyW9gyQKIMV1K0SZ-"
gdown.download(f"https://drive.google.com/uc?id={file_id}", quiet=False)

try:
    model.load_weights("gdown_model_weights.weights.h5")
except Exception as e:
    print(f"Could not load weights: {e}")
    print("Please ensure you have trained a model or that the weights file exists.")

# Prediction function
def predict_postfix(input_seq):
    # Pre-process input
    input_tokens = [char for char in input_seq if char.strip()]
    input_seq = preprocess_sequence(input_tokens, in_tokenizer)
    
    # Get initial states from encoder
    states = encoder_model.predict(np.expand_dims(input_seq, axis=0))
    
    # Generate initial target sequence
    target_seq = np.zeros((1, 1), dtype=np.int32)
    
    stop_condition = False
    decoded_sentence = []
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states
        )
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = out_tokenizer[sampled_token_index]
        
        if sampled_token == '[PAD]' or len(decoded_sentence) >= MAX_LEN:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_token)
        
        # Update target sequence and states
        target_seq = np.array([[sampled_token_index]])
        states = [h, c]
        
    return "".join(decoded_sentence)

# Example usage
test_expression = "a+b*c"
predicted_postfix = predict_postfix(test_expression)

print(f"Input: {test_expression}")
print(f"Predicted Postfix: {predicted_postfix}")