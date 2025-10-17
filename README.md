# Inside-the-MLP-A-Domain-Specific-Language-for-Mathematical-Exploration

https://github.com/user-attachments/assets/1e0a39a1-ce86-4ccf-aba0-3194f24792ff

This is a domain-specific language (DSL) for describing and visualizing multilayer perceptrons (MLPs) in an intuitive, human-readable form. It allows users to express network architectures using a simple symbolic syntax such as:

`mlp_byhand y = 5 | 4 -> 5 -> 7 -> 3 -> 2`

In this example, 5 represents the number of input features (e.g., `5` pixels per image), 4 denotes the number of samples (e.g., `4` images), and the numbers following the arrows specify the projected dimensions of each layer. For instance, `5 -> 7` means that the feature space of dimension 5 is linearly transformed into a space of dimension 7.

AIBYHand automatically visualizes the resulting MLP, showing the data matrices, network graph, and corresponding PyTorch code. The visualization reveals how each layer transforms the input features step by step, providing a clear view of the network’s internal computations.

This project bridges conceptual understanding and implementation by connecting symbolic representation, matrix visualization, and executable code. It is particularly useful for teaching, debugging, and exploring the mathematical structure behind neural networks — making abstract deep learning operations tangible and interactive.
