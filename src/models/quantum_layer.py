import torch
import torch.nn as nn
import pennylane as qml

class QuantumCircuitLayer(nn.Module):
    """
    PennyLane Variational Quantum Circuit (VQC) as a PyTorch module.
    Uses angle encoding for data and a hardware-efficient ansatz with 
    entangling layers.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 3, diff_method: str = 'backprop'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize trainable parameters for the variational circuit
        # 3 rotation angles (RX, RY, RZ) per qubit per layer
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.01)
        
        # Create the PennyLane device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Define the QNode
        @qml.qnode(self.dev, interface='torch', diff_method=diff_method)
        def _circuit(inputs, weights):
            # Angle Encoding: Encode classical data into quantum state
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational Ansatz
            for layer in range(self.n_layers):
                # Parameterized single-qubit rotations
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                
                # CNOT entangling ladder
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    
            # Measurement: Expectation value of PauliZ for all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        self.circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass processing a batch of inputs.
        
        Args:
            x: Input tensor, shape (batch_size, n_qubits)
            
        Returns:
            Output tensor, shape (batch_size, n_qubits)
        """
        # Ensure input is on CPU for PennyLane simulation
        x_cpu = x.cpu()
        
        # Process the batch by looping over samples
        batch_results = []
        for i in range(x_cpu.shape[0]):
            res = self.circuit(x_cpu[i], self.weights)
            # res is a tuple/list of scalar tensors, stack them
            batch_results.append(torch.stack(res))
            
        # Stack batch results and match input dtype (float32)
        out = torch.stack(batch_results).to(dtype=x.dtype, device=x.device)
        
        return out
