import perceval as pcvl
import numpy as np
from merlin import QuantumLayer
from merlin.core.merlin_processor import MerlinProcessor
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
from pprint import pprint
import random

# Python built-in random
random.seed(42)

# NumPy
np.random.seed(42)

# PyTorch CPU
torch.manual_seed(42)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X)          
    print("Preprocessing complete.")

    return X_train,y

X_train=pd.read_csv('data/credit_train.csv')
X_train,y_train = preprocess_data(X_train)

X_test=pd.read_csv('data/credit_test.csv')
X_test,y_test = preprocess_data(X_test)


X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

def create_quantum_circuit(m):
    """Create quantum circuit with specified number of modes

    Parameters:
    -----------
    m : int
        Number of quantum modes in the circuit

    Returns:
    --------
    pcvl.Circuit
        Complete quantum circuit with trainable parameters
    """
    m+=1
    # Left interferometer with trainable parameters
    wl = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_li{i}"))
        // pcvl.BS(),
        #// pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    # Variable phase shifters for input encoding
    c_var = pcvl.Circuit(m)
    for i in range(6):  # 6 input features
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - 4) // 2, pcvl.PS(px))

    # Right interferometer with trainable parameters
    wr = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_ri{i}"))
        // pcvl.BS(),
        #// pcvl.PS(pcvl.P(f"theta_ro{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    # Combine all components
    c = pcvl.Circuit(m)
    c.add(0, wl)
    c.add(0, c_var)
    c.add(0, wr)
    c.add(0, c_var)

    return c

def create_model(num_modes):
    m = num_modes
    no_bunching = False
    c = create_quantum_circuit(m)
    quantum_layer = QuantumLayer(
        input_size=6,
        circuit=c,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 0] * (m // 2) + [0] * (m % 2),
        no_bunching=no_bunching,
    )

    head = nn.Linear(quantum_layer.output_size, 2)
    return nn.Sequential(quantum_layer, head)
    
def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_epochs=50,
    batch_size=32,
    lr=0.02,
):
    """Train a model and track metrics

    Parameters:
    -----------
    model : nn.Module
        Model to train
    X_train, y_train : torch.Tensor
        Training data and labels
    X_test, y_test : torch.Tensor
        Test data and labels
    model_name : str
        Name for progress bar display
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for mini-batch training
    lr : float
        Learning rate

    Returns:
    --------
    dict
        Training results including accuracies, and final report
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    train_accuracies = []
    test_accuracies = []

    model.train()

    pbar = tqdm(range(n_epochs), leave=False, desc=f"Training the QNN")
    for _epoch in pbar:
        # Shuffle training data
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0

        # Mini-batch training
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (X_train.size()[0] // batch_size)
        losses.append(avg_loss)
        pbar.set_description(f"Training the QNN - Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(X_train)
            train_preds = torch.argmax(train_outputs, dim=1).numpy()
            train_acc = accuracy_score(y_train.numpy(), train_preds)
            train_accuracies.append(train_acc)

            # Test accuracy
            test_outputs = model(X_test)
            test_preds = torch.argmax(test_outputs, dim=1).numpy()
            test_acc = accuracy_score(y_test.numpy(), test_preds)
            test_accuracies.append(test_acc)

        model.train()

    # Generate final classification report
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test)
        final_test_preds = torch.argmax(final_test_outputs, dim=1).numpy()
        final_report = classification_report(y_test.numpy(), final_test_preds)

    return {
        "final_test_acc": test_accuracies[-1],
        "classification_report": final_report,}



QNN = create_model(6)
results = train_model(
                    QNN,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )

pprint(results)