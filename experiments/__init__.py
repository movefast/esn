from agents.columnar.columnar_agent_stable_v1_base import \
    RNNAgent as ESNAgentV2
from agents.gru_agent import RNNAgent as GRUAgent
from agents.rnn_agent import RNNAgent as RNNAgent

AGENT_DICT = {
    "GRU": GRUAgent,
    "ESN_V2": ESNAgentV2,
    "RNN": RNNAgent,
}
