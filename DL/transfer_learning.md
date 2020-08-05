# Transfer Learning

## Concept

Learning Task A has been trained with large amount of data; Now apply that
trained model to Task B, and hoping that it can:
* learn better
* learn faster


## Why transfer learning work?

Transfer learning works by the assumption that Task A and B carries enough
similarities such that the features or model learned from Task A are equally
useful to Task B. For example, training on generic object recognization are
applied to radiology images for cancer detection; generic speech recognization
model used for specific wake word (Andrew Ng)

## When does transfer learning make sense?

* Task A has a lot more training data than Task B.

- Task A and Task B have same input.

## How is it done?

One way of doing transferred learning is to take away the last layer in training model of A, along with
associated weights; Add a new layer with random weights. Re-train **only** the last
layer with training data from Task B.

