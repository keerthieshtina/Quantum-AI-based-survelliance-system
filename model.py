import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
NUM_QUBITS = 2
IMG_SIZE = 32
input_params = [Parameter(f"x{i}") for i in range(NUM_QUBITS)]
qc = QuantumCircuit(NUM_QUBITS)
for i in range(NUM_QUBITS):
    qc.ry(input_params[i], i)
for i in range(NUM_QUBITS - 1):
    qc.cx(i, i + 1)
sampler_qnn = SamplerQNN(
    circuit=qc,
    input_params=input_params,
    weight_params=[],
    sampler=Sampler()
)
q_layer = TorchConnector(sampler_qnn)
class QuantumAttention(nn.Module):
    def __init__(self, input_dim, num_qubits):
        super().__init__()
        self.qnn = q_layer
        self.fc_in = nn.Linear(input_dim, num_qubits)
        self.fc_out = nn.Linear(2 ** num_qubits, input_dim)
        self.norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        q_in = self.fc_in(x)
        q_out = self.qnn(q_in)
        out = self.fc_out(q_out)
        return self.norm(x + out)
class HybridQuantumTransformer(nn.Module):
    def __init__(self, n_qubits, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 8 * (IMG_SIZE // 4) * (IMG_SIZE // 4)
        self.q_attn = QuantumAttention(conv_out, n_qubits)
        self.dropout = nn.Dropout(0.7)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out, 16),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(16, n_classes)
        )
    def forward(self, x):
        x_conv = self.conv(x)
        x_attn = self.q_attn(x_conv)
        x = self.dropout(x_conv + x_attn)
        return self.classifier(x)