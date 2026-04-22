
# reference_policy.py
#
# Utility functions for managing the reference policy (π_ref) in CPL_pAOT.
# Isolated here so the reference policy implementation can be swapped out
# (e.g. replaced with BCO) without touching the main pAOT training logic.
#
# Lifecycle of the reference network in CPL_pAOT:
#   1. build_reference_network() constructs π_ref with the same architecture
#      as π_θ. Gradients are enabled so it can be BC-trained in phase 1.
#   2. During phase 1 (ref_bc_steps), π_ref is trained with BC directly.
#      π_θ (self.network) is not touched during this phase.
#   3. freeze_reference() is called at the end of phase 1. From this point
#      π_ref weights are fixed for the rest of training.
#   4. compute_reference_log_probs() is called during the pAOT phase to
#      evaluate log π_ref(a_t | s_t) for each timestep in a batch.
#
# To replace BC with a different reference policy approach (e.g. BCO):
#   - Train your alternative policy externally and save a checkpoint using
#     the standard Algorithm.save() format (key: 'network').
#   - Load it into reference_network manually before calling freeze_reference().
#   - compute_reference_log_probs() does not need to change.

import torch
from research.networks.base import ActorPolicy


# build_reference_network
# Constructs π_ref with the same architecture as π_θ.
# Gradients are enabled at construction so that phase 1 BC training works.
# freeze_reference() must be called after phase 1 to lock the weights.
def build_reference_network(observation_space, action_space, network_kwargs, device):
	reference_network = ActorPolicy(
		observation_space, action_space, **network_kwargs
	).to(device)
	return reference_network


# freeze_reference
# Disables all gradients on π_ref. Called once at the end of phase 1.
# After this call, reference_network weights are permanently fixed.
def freeze_reference(reference_network):
	for param in reference_network.parameters():
		param.requires_grad = False


# compute_reference_log_probs
# Evaluates log π_ref(a_t | s_t) for every timestep in a batch.
# Returns a tensor of shape (B, S) matching the current policy's lp tensor.
# Always called inside torch.no_grad() in CPL_pAOT._get_cpl_loss().
def compute_reference_log_probs(reference_network, obs, action):
	reference_network.eval()
	ref_dist = reference_network.actor(reference_network.encoder(obs))
	if isinstance(ref_dist, torch.distributions.Distribution):
		return ref_dist.log_prob(action)
	else:
		# deterministic policy: negative MSE as log-prob proxy
		# (matches the same approximation used in CPL and CPL_KL)
		assert ref_dist.shape == action.shape
		return -torch.square(ref_dist - action).sum(dim=-1)
