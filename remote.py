import perceval as pcvl
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.merlin_processor import MerlinProcessor
from merlin.measurement.strategies import MeasurementStrategy


remote_config = pcvl.RemoteConfig()
remote_config.set_token("YOUR TOKEN")
#remote_config.set_proxies({"https": "socks5h://USER:PASSWORD@HOST:PORT"})  # Optional proxy configuration
remote_config.save()

# 1) Create the Perceval RemoteProcessor (token must already be configured)
rp = pcvl.RemoteProcessor("qpu:ascella")

# 2) Wrap it with MerlinProcessor
proc = MerlinProcessor(
    rp,
    microbatch_size=32,        # batch chunk size per cloud call (<=32)
    timeout=3600.0,           # default wall-time per forward (seconds)
    max_shots_per_call=None,  # optional cap per cloud call (see below)
    chunk_concurrency=1       # parallel chunk jobs within a quantum leaf
)

# 3) Build a QuantumLayer and a small model
b = CircuitBuilder(n_modes=6)
b.add_rotations(trainable=True, name="theta")
b.add_angle_encoding(modes=[0, 1], name="px")
b.add_entangling_layer()

q = QuantumLayer(
    input_size=2,
    builder=b,
    n_photons=2,
    no_bunching=True,
    measurement_strategy=MeasurementStrategy.PROBABILITIES,  # raw probability vector
).eval()

model = nn.Sequential(
    nn.Linear(3, 2, bias=False),
    q,
    nn.Linear(15, 4, bias=False),   # 15 = C(6,2) from the chosen circuit
    nn.Softmax(dim=-1)
).eval()

# 4) Run remotely with sampling (nsample) or exact probs if available
X = torch.rand(8, 3)
y = proc.forward(model, X, nsample=5000)   # synchronous
print(y.shape)