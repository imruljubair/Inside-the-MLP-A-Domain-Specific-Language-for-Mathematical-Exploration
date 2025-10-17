import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from graphviz import Digraph
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from IPython.core.magic import register_line_magic
import pyperclip

def genLayers(a, batch=1):
    low, high = -1,1
    x = a.copy()
    y = []
    x.insert(0, batch)
    for i in range(1, len(x)):
        v=np.random.uniform(low, high, size=(x[i], x[i-1]))
        y.append(v)
    return y


def get_dimensions(dictionary):
    if not dictionary:
        return tuple()

    # Determine the number of dimensions by inspecting the first key
    num_dimensions = len(next(iter(dictionary)))

    # Initialize a list to hold the maximum index for each dimension
    max_indices = [0] * num_dimensions

    # Iterate through the keys to find the maximum index for each dimension
    for key in dictionary.keys():
        for dim in range(num_dimensions):
            if key[dim] > max_indices[dim]:
                max_indices[dim] = key[dim]

    # Add 1 to each maximum index to get the number of rows/columns/... in each dimension
    dimensions = tuple(max_index + 1 for max_index in max_indices)

    return dimensions


def pad_matrix_with_empty_strings(matrix, bottom_right=True):
    # Get the shape of the original matrix
    rows, cols = matrix.shape
    
    if bottom_right == True:
        # Create a new matrix with an extra row and column, filled with empty strings
        padded_matrix = np.full((rows + 1, cols + 1), " ", dtype=object)
    else:
        padded_matrix = np.full((rows + 1, cols), " ", dtype=object)
    
    # Copy the original matrix into the top-left corner of the new matrix
    padded_matrix[:rows, :cols] = matrix
    
    return padded_matrix


def processForBias(y):
    low, high = -1, 1
    rows, cols = y[0].shape
    ones_row = np.ones((1, cols))
    y[0] = np.concatenate((y[0], ones_row), axis=0)
    
    for i in range(1, len(y)):
        rows, cols = y[i].shape
#         ones_col = np.zeros((rows, 1))
        ones_col = np.random.uniform(low, high, size=(rows, 1))
        y[i] = np.concatenate((y[i], ones_col), axis=1)
    return y



def softmax(x):
    x = np.asarray(x)
    
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(shift_x)
    softmax_x = exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    return softmax_x



def createDataDict(M):
    y = M.copy()
    # y = processForBias(M.copy())
    
    matrix_dict = {}
#     matrix_dict[0,0] = np.eye(y[0].shape[0])
    matrix_dict[0,0] = np.full((y[0].shape[0], y[0].shape[0]), " ")
    matrix_dict[0,1] = y[0]
    
    for i in range(1, len(y)):
        
        matrix_dict[i, 0] = y[i]
        mm=matrix_dict[i,0] @ matrix_dict[i-1,1]
        if i < len(y)-1:
            mm = np.maximum(0, mm) # ReLU
            rows, cols = mm.shape
            ones_row = np.ones((1, cols))
            mm = np.concatenate((mm, ones_row), axis=0) # Add bias term
        else:
            mm = softmax(mm) # Softmax for the last layer
        matrix_dict[i, 1] = mm
        
    matrix_dict = {key: (value if i == 0 else np.round(value, 2)) for i, (key, value) in enumerate(matrix_dict.items())}

    return matrix_dict


def createLayoutDict(dataDict):
    layoutDict = {}
    p, _ = get_dimensions(dataDict)
    
    layoutDict[0,0] = dataDict[0,0]
    
    q = dataDict[0, 1]
    q_ = np.full(q.shape, "i")
    q_[-1, :] = "$"
    layoutDict[0,1] = q_
    
    for i in range(1, p):
        m = dataDict[i,0]
        m_ = np.full(m.shape, "W")
        m_[:, -1] = "b"
        layoutDict[i, 0] = m_
        
        n = dataDict[i,1]
        n_ = np.full(n.shape, "h")
        n_[-1, :] = "$"
        layoutDict[i,1] = n_
    
    # for j in range(1, p-1):
#         n = dataDict[j,1]
#         n_ = np.full(n.shape, "h")
#         n_[-1, :] = "$"
#         layoutDict[j,1] = n_
    
    r = dataDict[p-1, 1]
    r_ = np.full(r.shape, "o")
    layoutDict[p-1, 1] = r_
    
    return layoutDict



def getInfo(matrix_dict):
    global m,n,p
    left_matrix_cols = []
    right_matrix_cols = []
    right_matrix_rows = []

    for i,j in matrix_dict.keys():
        if j == 0:
            left_matrix_cols.append(matrix_dict[i,j].shape[1])
        if j == 1:
            right_matrix_cols.append(matrix_dict[i,j].shape[1])
            right_matrix_rows.append(matrix_dict[i,j].shape[0])


    m = np.max(left_matrix_cols)
    n = np.max(right_matrix_cols)
    p = np.sum(right_matrix_rows)
    
    return m,n,p



def sr(matrix_dict, i,j):
    if i == 0:
        return 0
    return sr(matrix_dict, i-1, j) + np.shape(matrix_dict[i-1, j])[0]

def sc(matrix_dict, i, j):
    if j == 0:
        return m - np.shape(matrix_dict[i, j])[1]
    return m


def insert_matrix(B, matrix_dict, i, j):
    # Get the dimensions of the smaller matrix S
    start_row = sr(matrix_dict, i,j)
    start_col = sc(matrix_dict, i,j)
    S = matrix_dict[i,j]
    rows_S, cols_S = S.shape

    # Ensure that S can fit inside B starting from (i, j)
    if start_row + rows_S > B.shape[0] or start_col + cols_S > B.shape[1]:
        raise ValueError("The smaller matrix S cannot fit into the larger matrix B at the specified position.")
    
    # Insert matrix S into matrix B starting at (i, j)
    B[start_row:start_row+rows_S, start_col:start_col+cols_S] = S
    
    return B


def fillCanvas(canvas_matrix, matrix_dict):
    for key in matrix_dict:
        canvas_matrix = insert_matrix(canvas_matrix, matrix_dict, key[0], key[1])
    return canvas_matrix


@register_line_magic
def mlp_byhand(line):
    parts = line.split('|')
    right_part = parts[1].strip()
    first_number_str = right_part.split()[0]
    batch = int(first_number_str) 
    parts = line.split('|')
    right_part = parts[1].strip()
    remaining_parts = right_part.split()[1:]
    new_right_part = ' '.join(remaining_parts)
    line = parts[0].strip() + ' ' + new_right_part
    var_name, sequence = line.split('=')
    var_name = var_name.strip()
    sequence = sequence.strip()
    
    elements = sequence.split('->')
    elements = [int(e.strip()) for e in elements]
    low, high = -1, 1
    x = elements.copy()
    y = []
    x.insert(0, batch)
    for i in range(1, len(x)):
        # v=np.random.uniform(low, high, size=(x[i], x[i-1]))
        if i == 1:
            # For the first matrix, use random values between 0 and 1
            v = np.random.uniform(0, 1, size=(x[i], x[i-1]))
        else:
            # For the rest of the matrices, use random values between -1 and 1
            v = np.random.uniform(-1, 1, size=(x[i], x[i-1]))
        y.append(v)
    get_ipython().user_ns[var_name] = y
    
    
@register_line_magic
def mlp_byhand_new(line):
    ipython = get_ipython()
    
    # Extract the variable name and the sequence of layer sizes
    var_name, sequence = line.split('=')
    var_name = var_name.strip()
    sequence = sequence.strip()

    # Parse the sequence to get the initial variable and layer sizes
    elements = sequence.split('->')
    initial_var_name = elements[0].strip()
    elements = [int(e.strip()) for e in elements[1:]]

    # Define the range for weight initialization
    low, high = -1, 1

    # Create the list of layer sizes including the initial variable size
    y = []

    # Check if the provided initial variable name exists in the namespace
    if initial_var_name in ipython.user_ns:
        # Use the existing variable as the initial weight matrix
        y.append(ipython.user_ns[initial_var_name])
        input_size = y[0].shape[0]
    else:
        raise ValueError(f"Variable {initial_var_name} is not defined in the current namespace.")

    # Generate weight matrices for each subsequent layer transition
    for output_size in elements:
        v = np.random.uniform(low, high, size=(output_size, input_size))
        y.append(v)
        input_size = output_size

    # Store the weight matrices in the notebook's namespace
    ipython.user_ns[var_name] = y

@register_line_magic
def mlp_byhand_model(line):
    # get the variable name from the input line
    #     var_name = line.strip()

    #     # get the model from the variable name
    #     model = get_ipython().user_ns[var_name]
    var_name, num = line.strip().split('|')
    num = num.strip()
    var_name = var_name.rstrip()

    # get the model from the variable name
    model = get_ipython().user_ns[var_name]

    # extract the architecture information from the model
    layers = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            layers.append(layer.out_features)

    # get the number of input features from the first linear layer
    input_features = model[0].in_features

    # format the output in the specified way
    output = f'y = {input_features} | {num} -> ' + ' -> '.join(map(str, layers))
    # print(input_features)
    # print(layers)
    # print("Copy the following commands and run in a cell:\n")
    # print("%mlp_byhand " + output + "\n" + "reveal(y)")
    dsl = f'%mlp_byhand {output}\nreveal(y)'
    print(f'## Copy the following commands and run in a cell:\n\n{dsl}')
    # visForDSL(dsl)


def createMetaDataDict(dataDict):
    p,q = get_dimensions(dataDict)
    MetaDataDict = {}
    MetaDataDict[0,0] = "empty" # tope left matrix is empty
    MetaDataDict[0,1] = "Input" # first right matrix are the input
    
    for i in range(1, p):
        MetaDataDict[i, 0] = f"W_{i}" # left matrices are weights and biases
        MetaDataDict[i, 1] = f"ReLU(Layer{i})" # right matrices are hidden layers
        
    MetaDataDict[p-1,q-1] = "Softmax(Output)"
    return MetaDataDict
        
        


# In[17]:


def padDownArrowRightDict(dataDict):
    paddedDict = dataDict.copy()
    p,q = get_dimensions(dataDict)
    
    for i in range(p-1):
        right_rows, right_cols = dataDict[i, 1].shape
        padded_matrix = np.full((right_rows+1, right_cols), "↓", dtype=object)
        padded_matrix[:right_rows, :right_cols] = dataDict[i, 1]
        paddedDict[i,1] = padded_matrix
        
    return paddedDict


def padRightArrowLeftDict(dataDict):
    paddedDict = dataDict.copy()
    p,q = get_dimensions(dataDict)
    
    for i in range(1,p):
        left_rows, left_cols = dataDict[i, 0].shape
        padded_matrix = np.full((left_rows, left_cols+1), "→", dtype=object)
        padded_matrix[:left_rows, :left_cols] = dataDict[i, 0]
        paddedDict[i,0] = padded_matrix
        
    return paddedDict



def padRowLeftDict(dataDict):
    paddedDict = dataDict.copy()
    p,q = get_dimensions(dataDict)
    
    for i in range(p):
        left_rows, left_cols = dataDict[i, 0].shape
        right_rows, right_cols = dataDict[i, 1].shape
        padded_matrix = np.full((right_rows, left_cols), " ", dtype=object)
        padded_matrix[:left_rows, :left_cols] = dataDict[i, 0]
        paddedDict[i,0] = padded_matrix
    
    
    
    return paddedDict



def addLabelsLayers(dataDict):
    paddedDict = dataDict.copy()
    labels = createMetaDataDict(dataDict)
    p,q = get_dimensions(dataDict)
    
    for i in range(p):
        right_rows, right_cols = dataDict[i, 1].shape
        padded_matrix = np.full((right_rows, right_cols + 1), " ", dtype=object)
        padded_matrix[:right_rows, :right_cols] = dataDict[i, 1]
        # padded_matrix[0, -1] = f" 👈 {labels[i,1]}"
        padded_matrix[0, -1] = ""
        paddedDict[i,1] = padded_matrix
       
    return paddedDict
    


def getStatsFromObjMatrix(matrix):
    numeric_values = np.array([x for row in matrix for x in row if isinstance(x, (int, float))])
    # Calculate min, max, and std
    min_value = numeric_values.min()
    max_value = numeric_values.max()
    mean_value = numeric_values.mean()
    std_value = numeric_values.std()
    

    # Output the results
    return min_value, max_value, mean_value, std_value       



def colorize7(val_from_df1, val_from_df2):
    if val_from_df2 == 'i':
        val = np.float16(val_from_df1)
        # alpha = 1.0 / (1.0 + np.exp(-val))
        # alpha = ((val+1)/2)**(1.0/2.0)
        alpha = val**0.5
        return f'background-color: rgba(204, 201, 35, {(alpha):.2f}); border: 0px solid orange;'
    
    elif val_from_df2 == 'W':
        val = np.float16(val_from_df1)
        alpha = 1.0 / (1.0 + np.exp(-(val - mean_value) / std_value))
        return f'background-color: rgba(45, 180, 45, {alpha:.2f}); border: 0px solid green;'
    
    elif val_from_df2 == 'h':
        val = np.float16(val_from_df1)
        alpha = 1.0 / (1.0 + np.exp(-(val - mean_value) / std_value))
        return f'background-color: rgba(65, 136, 247, {alpha:.2f}); border: 0px solid blue;'
    
    elif val_from_df2 == '$':
        return f'color: rgba(160, 160, 160, 1); background-color: white; border: 0px solid white;'
        # return f'color: rgba(255, 255, 255, 1); background-color: rgba(220, 220, 220, 1); border: 1px solid white;'

    elif val_from_df2 == 'b':
        val = np.float16(val_from_df1)
        alpha = 1.0 / (1.0 + np.exp(-(val - mean_value) / std_value))
        return f'background-color: rgba(180, 55, 55, {alpha:.2f}); border: 0px solid red;'
    
    elif val_from_df2 == 'o':
        val = np.float16(val_from_df1)
        # alpha = 1.0 / (1.0 + np.exp(-(val - mean_value) / std_value))
        # alpha = val**(1.0/2.0)
        alpha = val**0.5
        return f'background-color: rgba(171, 116, 179, {alpha:.2f}); border: 0px solid purple;'
    
    return 'background-color: white; border: 0px solid white;'


def createCanvas(y):
    
    # y = processForBias(y.copy())
    
    matrix_dict = createDataDict(y.copy())

    matrix_dict_ = createLayoutDict(matrix_dict)

    downArrowDict = padDownArrowRightDict(matrix_dict)
    downArrowDict_ = padDownArrowRightDict(matrix_dict_)

    rightArrowDict = padRightArrowLeftDict(downArrowDict)
    rightArrowDict_ = padRightArrowLeftDict(downArrowDict_)

    padLeftDict = padRowLeftDict(rightArrowDict)
    padLeftDict_ = padRowLeftDict(rightArrowDict_)


    labeledDict = addLabelsLayers(padLeftDict)
    labeledDict_ = addLabelsLayers(padLeftDict_) 
    
    m, n, p = getInfo(labeledDict)
    canvas_matrix = np.full((p+1, m+n+1), " ", dtype=object)
    canvas_matrix_ = np.full((p+1, m+n+1), " ", dtype=object)

    canvas_matrix = fillCanvas(canvas_matrix, labeledDict)
    canvas_matrix_ = fillCanvas(canvas_matrix_, labeledDict_)
    
    return canvas_matrix, canvas_matrix_

def countFeatures(y):
    feats = []
    for k in y:
        feats.append(k.shape[0])
    batch = y[0].shape[1]
    return feats, batch



def getCode_old(y):
    # Define the sizes of the layers
    layer_sizes = countFeatures(y) # Example sizes: input_size, hidden_size1, hidden_size2, output_size

    # Define the HTML-formatted Python code to be displayed in the text box
    code = """import torch\nimport torch.nn as nn\n"""

    # Add layer sizes to the code string
    code += f"input_size = {layer_sizes[0]}\n"

    # Add the MLP creation code
    code += """mlp = nn.Sequential(\n"""
    # Add layers to the code string
    for i in range(len(layer_sizes) - 1):
        code += f"        nn.Linear({layer_sizes[i]}, {layer_sizes[i + 1]}),\n"
        if i < len(layer_sizes) - 2:  # Add ReLU activation for all layers except the last one
            code += "        nn.ReLU( ),\n"

    # Add Softmax layer at the end
    code += "        nn.Softmax(dim=1)\n"

    # Close the Sequential block
    code += """)\n# Create a random input tensor\ninput_tensor = torch.randn(1, input_size)\n
# Perform a forward pass\noutput_tensor = mlp(input_tensor)\n
# Print the output\nprint(output_tensor)"""

    return code


def getCode(y):
    # Define the sizes of the layers
    layer_sizes, batch = countFeatures(y) # Example sizes: input_size, hidden_size1, hidden_size2, output_size

    # Define the HTML-formatted Python code to be displayed in the text box
    code = """import torch\nimport torch.nn as nn\n\n"""

    # Add layer sizes to the code string
    code += f"input_size = {layer_sizes[0]}\n"
    code += f"output_size = {layer_sizes[-1]}\n"
    code += f"batch_size = {batch}\n\n"
    
    # Add the MLP creation code
    code += """mlp = nn.Sequential(\n"""
    # Add layers to the code string
    for i in range(len(layer_sizes) - 1):
        if i == 0:
            code += f"        nn.Linear(input_size, {layer_sizes[i + 1]}),\n"
        elif i == len(layer_sizes) - 2:
            code += f"        nn.Linear({layer_sizes[i]}, output_size),\n"
        else:
            code += f"        nn.Linear({layer_sizes[i]}, {layer_sizes[i + 1]}),\n"
            if i < len(layer_sizes) - 2:  # Add ReLU activation for all layers except the last one
                code += "        nn.ReLU(),\n"

    # Add Softmax layer at the end
    code += "        nn.Softmax(dim=1)\n)\n\n"

#     # Close the Sequential block
#     code += """)\n# Create a random input tensor\ninput_tensor = torch.randn(1, input_size)\n
# # Perform a forward pass\noutput_tensor = mlp(input_tensor)\n
# # Print the output\nprint(output_tensor)"""
    
    code += """# Create a random input tensor\n"""
    code += """input_tensor = torch.randn(batch_size, input_size)\n\n"""
    code += """# Perform a forward pass\n"""
    code += """output_tensor = mlp(input_tensor)\n\n"""
    code += """print(output_tensor)"""
 
    return code



def changeWb(y):
    low, high = -1, 1
    y_new_wb = y.copy()
    for i in range(1, len(y)):
        z = np.random.uniform(low, high, size=y[i].shape)
        y_new_wb[i] = z
    return y_new_wb


def changeX(y):
    low, high = -1, 1
    y_new_x = y.copy()
    # for i in range(1, len(y)):
    #     z = np.random.uniform(low, high, size=y[i].shape)
    #     y_new_wb[i] = z
    z = np.random.uniform(low, high, size=y[0].shape)
    y_new_x[0] = z
    return y_new_x



def doVis(styled_df, y):
    output1 = widgets.Output()
    output2 = widgets.Output()
    output3 = widgets.Output()

    with output1:
        display(styled_df)

    with output2:
        layer_sizes, batch = countFeatures(y)
        dot = createGraph(layer_sizes)
        display(dot)

    with output3:
        code = getCode(y)
        displayCodeWithCopyButton(code)

    output1_scroll = createScrollableBox(output1)
    output2_scroll = createScrollableBox(output2)
    output3_scroll = createScrollableBox(output3)

    grid1 = createGridBox([output1_scroll])
    grid2 = createGridBox([output1_scroll, output2_scroll])
    grid3 = createGridBox([output1_scroll, output2_scroll, output3_scroll])

    tab = createTabWidget([grid1, grid2, grid3])
    display(tab)

def createGraph(layer_sizes):
    dot = Digraph()
    common_node_attrs = {
        'shape': 'point',
        'style': 'filled',
        'width': '0.15',
        'height': '0.15',
        'fontsize': '1',
        'color': 'gray'
    }
    edge_attrs = {
        'penwidth': '0.4',
        'arrowhead': 'none',
        'color': 'gray',
        'label': ''
    }

    for layer_idx, layer_size in enumerate(layer_sizes):
        for node_idx in range(layer_size):
            node_id = f'L{layer_idx}N{node_idx}'
            fillcolor = getNodeColor(layer_idx, layer_sizes)
            dot.node(node_id, fillcolor=fillcolor, **common_node_attrs)

    for layer_idx in range(len(layer_sizes) - 1):
        for src_node_idx in range(layer_sizes[layer_idx]):
            for dst_node_idx in range(layer_sizes[layer_idx + 1]):
                src_node_id = f'L{layer_idx}N{src_node_idx}'
                dst_node_id = f'L{layer_idx + 1}N{dst_node_idx}'
                dot.edge(src_node_id, dst_node_id, **edge_attrs)

    return dot

def getNodeColor(layer_idx, layer_sizes):
    if layer_idx == 0:
        return 'gold'
    elif layer_idx == len(layer_sizes) - 1:
        return 'orchid'
    else:
        return 'skyblue2'

def displayCodeWithCopyButton(code):
    text_box = widgets.Textarea(
        value=code,
        placeholder='Type here',
        disabled=False,
        layout=widgets.Layout(height='300px')
    )
    copy_button = widgets.Button(
        description='copy code',
        icon='clipboard',
        layout=widgets.Layout(width='80px')
    )

    def copy_to_clipboard(b):
        pyperclip.copy(text_box.value)

    copy_button.on_click(copy_to_clipboard)
    # display(widgets.VBox([copy_button, text_box]))
    display(text_box)

def createScrollableBox(output):
    return widgets.Box(children=[output], layout=widgets.Layout(border='none', overflow_x='scroll', width='100%'))

def createGridBox(children):
    num_children = len(children)
    column_width = str(100 / num_children) + "%"

    grid = widgets.GridBox(children=children, layout=widgets.Layout(grid_template_columns=column_width * num_children))
    return grid
    # return widgets.GridBox(children=children, layout=widgets.Layout(grid_template_columns="33.33%" * len(children)))
    # return widgets.GridBox(children=children, layout=widgets.Layout(grid_template_columns=str(100.00/len(children))+"%" * len(children)))


def createTabWidget(children):
    tab = widgets.Tab()
    tab.children = children
    tab.set_title(0, 'Matrix')
    tab.set_title(1, 'Graph')
    tab.set_title(2, 'Code')
    return tab



def visForDSL(dsl):
    output = widgets.Output()
    with output:
        displayDSLWithCopyButton(dsl)
    # output_scroll = createScrollableBox(output)
    display(output)
    

def reveal(y):
    # Create outputs only once
    matrix_output = widgets.Output()
    graph_output = widgets.Output()
    code_output = widgets.Output()
    
    # Create DataFrames and widgets only once
    with matrix_output:
        y_ = processForBias(y.copy())
        canvas_matrix, canvas_matrix_ = createCanvas(y_)
        styled_df_widget = widgets.HTML()  # Create an HTML widget for the styled DataFrame
        initial_df = createDataframe(canvas_matrix, canvas_matrix_)[2]._repr_html_()
        styled_df_widget.value = initial_df
        display(styled_df_widget)
    
    with graph_output:
        layer_sizes, batch = countFeatures(y)
        dot = createGraph(layer_sizes)
        display(dot)
    
    with code_output:
        code_widget = widgets.Textarea(
            disabled=False,
            layout=widgets.Layout(height='300px')
        )
        copy_button = widgets.Button(
            description='copy code',
            icon='clipboard',
            layout=widgets.Layout(width='80px')
        )
        
        def copy_to_clipboard(b):
            pyperclip.copy(code_widget.value)
        
        copy_button.on_click(copy_to_clipboard)
        code_widget.value = getCode(y)
        # display(widgets.VBox([copy_button, code_widget]))
        display(code_widget)
    
    def update_outputs(y):
        # Update only the data, not the widgets
        y_ = processForBias(y.copy())
        canvas_matrix, canvas_matrix_ = createCanvas(y_)
        new_df = createDataframe(canvas_matrix, canvas_matrix_)[2]._repr_html_()
        styled_df_widget.value = new_df
        
        # Update code without recreating widgets
        code_widget.value = getCode(y)
        
        # Graph needs to be redrawn due to Graphviz limitations
        with graph_output:
            graph_output.clear_output(wait=True)
            layer_sizes, batch = countFeatures(y)
            dot = createGraph(layer_sizes)
            display(dot)
    
    # Create a button to randomize the matrices
    button = widgets.Button(
        description='Randomize Matrices',
        layout=widgets.Layout(width='150px', margin='10px 10px'),
        # button_style='background-color: #4CAF50; color: white; font-size: 16px;'
    )
    
    def randomize_matrices(button):
        nonlocal y
        y = changeWb(y)
        update_outputs(y)
    
    button.on_click(randomize_matrices)
    
    # Create containers
    matrix_container = widgets.VBox([
        button,
        matrix_output
    ])
    
    # Create scrollable boxes
    matrix_scroll = createScrollableBox(matrix_container)
    graph_scroll = createScrollableBox(graph_output)
    code_scroll = createScrollableBox(code_output)
    
    # Create grid boxes
    grid1 = createGridBox([matrix_scroll])
    grid2 = createGridBox([matrix_scroll, graph_scroll])
    grid3 = createGridBox([matrix_scroll, graph_scroll, code_scroll])
    
    # Create and display tab
    tab = createTabWidget([grid1, grid2, grid3])
    display(tab)

def createGraph(layer_sizes):
    dot = Digraph()
    common_node_attrs = {
        'shape': 'point',
        'style': 'filled',
        'width': '0.15',
        'height': '0.15',
        'fontsize': '1',
        'color': 'gray'
    }
    edge_attrs = {
        'penwidth': '0.4',
        'arrowhead': 'none',
        'color': 'gray',
        'label': ''
    }
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        for node_idx in range(layer_size):
            node_id = f'L{layer_idx}N{node_idx}'
            fillcolor = getNodeColor(layer_idx, layer_sizes)
            dot.node(node_id, fillcolor=fillcolor, **common_node_attrs)
    
    for layer_idx in range(len(layer_sizes) - 1):
        for src_node_idx in range(layer_sizes[layer_idx]):
            for dst_node_idx in range(layer_sizes[layer_idx + 1]):
                src_node_id = f'L{layer_idx}N{src_node_idx}'
                dst_node_id = f'L{layer_idx + 1}N{dst_node_idx}'
                dot.edge(src_node_id, dst_node_id, **edge_attrs)
    
    return dot


def createDataframe(canvas_matrix, canvas_matrix_):
    global min_value, max_value, mean_value, std_value
    min_value, max_value, mean_value, std_value = getStatsFromObjMatrix(canvas_matrix)

    df1 = pd.DataFrame(canvas_matrix.astype(str)) 
    df2 = pd.DataFrame(canvas_matrix_.astype(str)) 
    df_style = [
    # {'selector': 'table', 'props': [
    #                     ('border-collapse', 'collapse'),
    #                     ('font-family', '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'),
    #                     ('font-size', '14px'),
    #                     ('border', '1px solid #ddd')
    #                 ]},
        {
            'selector': 'table',
            'props': [
                ('table-layout', 'flex'),  # Set the table layout to fixed
                ('width', '100%'),  # Set the width of the table to 100%
                # ('border-spacing', '0'),
                # ('border-collapse', 'collapse')
            ]
        },
            {
                'selector': 'th',
                'props': [('display', 'none')]
            },
            {
                'selector': 'td',
                'props': [
                    ('font-size', '5.5pt'),
                    ('text-align', 'center'),
                    # ('height', '2px'),
                    # ('width', '30px'),   # Minimum column width
                    # ('max-width', '10px'),  # Maximum column width
                    ('text-overflow', 'ellipsis'),
                    # ('white-space', 'nowrap'),  # Prevent text wrapping
                    ('padding', '1.8px'),  # Reduce the padding to 2px
                    ('line-height', '1.5'),
                    # ('line-width', '10') 
                ]

            },

        # {
        #         'selector': 'tr',
        #         'props': [
        #             ('max-height', '3px'),
        #             ('min-height', '2px'),
        #             ('white-space', 'nowrap')
        #         ]
        # },
        {
            'selector': 'td:hover',  # Hover effect for cells
            'props': [
                ('background-color', 'white'),  # Light khaki on hover
                ('cursor', 'pointer'),            # Pointer cursor for clickable feel
                ('font-weight', 'bold'),
                # ('font-size', '6pt'),
            ]
        },
        ]

    styled_df = df1.style\
                        .apply(lambda x: np.vectorize(colorize7)(df1, df2), axis=None)\
                        .set_table_styles(df_style)
    
    styled_df.to_html('/content/styled_df.html')
    return df1, df2, styled_df

