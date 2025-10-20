# Inside the MLP: A Domain Specific Language for Mathematical Exploration
This is a prototype of a domain-specific language (DSL) that helps users *see and understand* the inner mathematics of multilayer perceptrons (MLPs) through **matrix representations and visual projections**.

<img width="1329" height="638" alt="image" src="https://github.com/user-attachments/assets/67fa8f50-0857-4203-8eae-3787cd3690ed" />

# 🧠 Visualizing MLPs Through Matrices and Code

Using a simple syntax such as:

`mlp_byhand y = 5 | 4 -> 5 -> 7 -> 3 -> 2`

Users can describe the data flow and layer transformations of an MLP.

- **5** → number of input features (e.g., pixels of an image).  
- **4** → number of samples (e.g., images in a batch).  
- Each arrow `->` defines how one feature space is linearly projected into another (e.g., `5 -> 7` transforms a 5-dimensional feature space into 7 dimensions through a weight matrix).

![abhdsl](https://github.com/user-attachments/assets/058535f0-1ae0-44c4-b8bf-8ba9028282f1)

The program automatically generates interactive visualizations showing:

- 🧩 **Matrix views** — each layer’s transformation represented by color-coded weight matrices and resulting activations. It utilizes **Prof. Tom Yeh**'s framework to present the maths behind AI, widely knowns as [AI by Hand](https://www.byhand.ai/).
- 🔗 **Network graph** — a structural view of neurons and their connections.  
- 💻 **Code view** — the equivalent PyTorch implementation for direct experimentation.

---

## 🎯 How to use

Open this [Google Colab](https://colab.research.google.com/drive/1XLMQeNpOJspad47W8YPoPhWPC8au0arg?usp=sharing) and play! See this tutorial video.

https://github.com/user-attachments/assets/1e0a39a1-ce86-4ccf-aba0-3194f24792ff

---

## 🎯 Who Is This For

- **Students** exploring how MLPs transform data through matrix multiplications.  
- **Educators** seeking visual, interactive ways to teach linear layers and activations.  
- **Researchers & developers** wanting to debug or introspect the hidden computations of small neural networks.

---

## 🚀 Why It Matters

Neural networks are often treated as black boxes. The work opens that box by showing how every linear projection, activation, and feature transformation is realized through matrices.  
It makes abstract math *visible*, helping users connect deep learning theory with actual data flow.

By bridging **symbolic representation → matrix computation → executable PyTorch code**,  
This transforms the learning of MLPs from a theoretical concept into a *hands-on, visual experience*.
