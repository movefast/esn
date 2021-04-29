from agents.columnar.columnar_agent_stable_v1_base import \
    RNNAgent as ESNAgentV2
from agents.gru_agent import RNNAgent as GRUAgent
from agents.rnn_agent import RNNAgent as RNNAgent
from agents.trace_agent import RNNAgent as TraceAgent

AGENT_DICT = {
    "GRU": GRUAgent,
    "ESN_V2": ESNAgentV2,
    "RNN": RNNAgent,
    "Trace": TraceAgent,
}

AGENT_PARAMS = {
    "RNN": {"step_size": 0.0025},
    "GRU": {"step_size": 0.0025},
    "SubsampleAgent": {"step_size": 0.005},
    "ESN_V2": {"step_size": 0.00125, "beta":0.15},
    "Trace": {"step_size": 0.00125, 'alpha':0.1},
}
