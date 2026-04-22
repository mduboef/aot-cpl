
# reference_policy.py
#
# This file is intentionally isolated from cpl_paot.py so that the reference
# policy implementation can be swapped out without touching the main pAOT
# training logic. To use a different reference policy (e.g. BCO, offline RL):
#   1. Write a new function analogous to build_reference_network() that
#      produces a trained ActorPolicy checkpoint.
#   2. Replace the call to snapshot_reference() in CPL_pAOT.train_step() with
#      your new loading logic.
#   3. compute_reference_log_probs() does not need to change -- it only
#      depends on the frozen ActorPolicy interface.
#
# Current implementation: BC on all segments in the preference dataset.
#   - The reference network is initialized with the same architecture as the
#     current policy (ActorPolicy).
#   - During CPL_pAOT's bc_steps pretraining phase, the current network is
#     trained with the BC objective.
#   - At the end of bc_steps, snapshot_reference() copies the BC-trained
#     weights into the frozen reference network.
#   - From that point on, compute_reference_log_probs() is called with
#     torch.no_grad() to produce the reference margins used in pAOT step 4.
#
# Potential issue with simple BC reference:
#   The preference dataset contains behavior from the data-collection policy,
#   not clean expert demonstrations. BC on this data may produce a mediocre
#   reference policy. If reference margins v_ref are poorly calibrated (e.g.,
#   consistently near zero because the reference cannot distinguish segments),
#   the pAOT OT matching in step 5 may not provide a useful training signal.
#
# If BC reference underperforms, consider:
#   - BCO (Behavioral Cloning from Observations): uses inverse dynamics model
#     to infer actions from state-only demonstrations. More robust when the
#     demonstration data is imperfect.
#   - IQL or CQL pre-training: trains a Q-function on the preference data to
#     produce a stronger reference policy.
#   - Using a subset of only the preferred segments for BC (bc_data="pos").

import torch
from research.networks.base import ActorPolicy


# build_reference_network
# Creates a frozen ActorPolicy with the same architecture as the main network.
# Called once during CPL_pAOT.__init__ / setup_network.
# The returned module has requires_grad=False on all parameters.
def build_reference_network(observation_space, action_space, network_kwargs, device):
	reference_network = ActorPolicy(
		observation_space, action_space, **network_kwargs
	).to(device)
	# freeze all parameters -- reference policy is never updated via gradients
	for param in reference_network.parameters():
		param.requires_grad = False
	return reference_network


# snapshot_reference
# Copies the current (BC-trained) network weights into the frozen reference.
# Called exactly once: at the end of the BC pretraining phase in train_step().
# After this point, reference_network weights are fixed for the entire CPL phase.
def snapshot_reference(reference_network, source_network):
	reference_network.encoder.load_state_dict(source_network.encoder.state_dict())
	reference_network.actor.load_state_dict(source_network.actor.state_dict())
	# keep frozen -- no requires_grad=True is set here intentionally


# compute_reference_log_probs
# Evaluates log π_ref(a_t | s_t) for each (obs, action) timestep in a batch.
# Returns a tensor of shape (B, S) matching the current policy's lp tensor.
# Always called inside torch.no_grad() by CPL_pAOT._get_cpl_loss().
#
# This is the only function that needs to change if you replace BC with BCO:
# swap the ActorPolicy forward pass for your alternative model's log_prob method.
def compute_reference_log_probs(reference_network, obs, action):
	reference_network.eval()
	ref_dist = reference_network.actor(reference_network.encoder(obs))
	if isinstance(ref_dist, torch.distributions.Distribution):
		# stochastic policy: true log probability
		return ref_dist.log_prob(action)
	else:
		# deterministic policy: use negative MSE as a proxy for log-prob
		# (matches the same approximation used in CPL and CPL_KL)
		assert ref_dist.shape == action.shape
		return -torch.square(ref_dist - action).sum(dim=-1)
