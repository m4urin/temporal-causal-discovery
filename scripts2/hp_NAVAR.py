from src.data.dataset import toy_data_3_nodes
from src.models.implementations.navar import NAVAR
from src.training.hyper_optimization import run_hyperopt

dataset = toy_data_3_nodes(time_steps=500)

architectures = [NAVAR]

evaluation_results = run_hyperopt(dataset, architectures, max_evals=20)

print(evaluation_results)
