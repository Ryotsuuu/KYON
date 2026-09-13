"""
agents/NN
=========
Neural Network Hive-Mind & Policy-Value Network Subsystem.
"""
from agents.NN.policy_value_net import (
    HiveMindPolicyValueNet,
    get_hive_mind_net,
    encode_dynamic_state,
    STATE_DIM,
    ACTION_DIM,
)

# Alias for backward compatibility
PolicyValueNet = HiveMindPolicyValueNet
encode_state = encode_dynamic_state

__all__ = [
    'HiveMindPolicyValueNet',
    'PolicyValueNet',
    'get_hive_mind_net',
    'encode_dynamic_state',
    'encode_state',
    'STATE_DIM',
    'ACTION_DIM',
]
