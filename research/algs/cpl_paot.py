
# cpl_paot.py
#
# CPL with Paired Alignment via Optimal Transport (pAOT).
# Implements the algorithm described in aot_cpl.tex, section:
#   "CPL with pAOT"
#
# Key difference from uAOT:
#   uAOT sorts raw preferred/rejected segment scores independently.
#   pAOT instead operates on per-pair MARGINS (preferred_score - rejected_score)
#   computed under both the current policy π_θ and a frozen reference policy π_ref.
#   Both margin sets are sorted independently, then OT-matched and compared.
#
# Why margins instead of raw scores?
#   Margins normalize away absolute log-prob scale differences, focusing the
#   model on relative preference discrimination. Comparing policy margins to
#   reference margins creates a distributional anchor: the policy is pushed to
#   produce better within-pair discrimination than the reference policy, not
#   just to maximize preferred-segment log-probs absolutely.
#
# Reference policy (tex step 3):
#   Obtained via BC on all segments in the preference dataset.
#   Reference network weights are frozen after the BC pretraining phase.
#   All reference policy logic is isolated in reference_policy.py so it can
#   be replaced with BCO or another approach without changing this file.
#   See reference_policy.py for details on the BC approach and its limitations.
#
# Data mode requirement:
#   FeedbackBuffer must use mode="comparison" so that each batch contains
#   (obs_1, action_1, obs_2, action_2, label). Per-pair margins require
#   knowing which segment in each pair is preferred vs. rejected.
#
# Discount factor note:
#   The formalization defines score_π(σ) = Σ_t γ^t α log π(a_t|s_t).
#   Baseline CPL does NOT apply γ^t (equivalent to γ=1). We match that
#   convention here for consistency. The FeedbackBuffer provides a
#   "discount" key in the batch but it is intentionally not used.

import itertools
from typing import Any, Dict, Type

import torch

from research.networks.base import ActorPolicy
from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm
from .reference_policy import (
	build_reference_network,
	compute_reference_log_probs,
	snapshot_reference,
)


# paot_loss
# Formalization reference: Steps 5 and 6 of "CPL with pAOT"
#
# Inputs:
#   u_theta -- policy margins per preference pair, shape (n,)
#              u_θ^i = score(σ_i+; π_θ) - score(σ_i-; π_θ)  (tex step 4)
#   v_ref   -- reference margins per preference pair, shape (n,)
#              v_ref^i = score(σ_i+; π_ref) - score(σ_i-; π_ref)  (tex step 4)
#
# Step 5 (tex): Sort policy margins and reference margins independently.
#   u_θ^(1) ≤ u_θ^(2) ≤ ... ≤ u_θ^(n)
#   v_ref^(1) ≤ v_ref^(2) ≤ ... ≤ v_ref^(n)
#   The i-th lowest policy margin is matched with the i-th lowest reference margin.
#
# Step 6 (tex): CPL loss on OT-matched margin pairs.
#   L(θ) = (1/n) Σ_i -log [ exp(u_θ^(i)) / (exp(u_θ^(i)) + exp(v_ref^(i))) ]
#
# Interpretation: the loss pushes the policy to achieve a higher within-pair
# margin than the reference policy achieves, after OT re-matching.
def paot_loss(u_theta, v_ref):
	# step 5: sort policy margins and reference margins independently
	u_sorted = torch.sort(u_theta).values  # u_θ^(1) ≤ ... ≤ u_θ^(n)
	v_sorted = torch.sort(v_ref).values    # v_ref^(1) ≤ ... ≤ v_ref^(n)

	# step 6: compute loss on OT-matched margin pairs
	# logit = u_sorted[i] - v_sorted[i]  (positive when policy margin > reference margin)
	logit = u_sorted - v_sorted

	# numerically stable BCE: -log sigmoid(logit) = log(1 + exp(-logit))
	max_val = torch.clamp(-logit, min=0)
	loss = (torch.log(torch.exp(-max_val) + torch.exp(-logit - max_val)) + max_val).mean()

	# accuracy: fraction of matched pairs where policy margin > reference margin
	with torch.no_grad():
		accuracy = (u_sorted > v_sorted).float().mean()

	return loss, accuracy


class CPL_pAOT(OffPolicyAlgorithm):
	def __init__(
		self,
		*args,
		alpha: float = 1.0,
		bc_coeff: float = 0.0,
		bc_data: str = "all",
		bc_steps: int = 10000,
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		assert "encoder" in self.network.CONTAINERS
		assert "actor" in self.network.CONTAINERS
		# pAOT requires bc_steps > 0 so the reference policy has been BC-trained
		# before the CPL phase starts (formalization step 3)
		assert bc_steps > 0, (
			"CPL_pAOT requires bc_steps > 0. The reference policy is obtained "
			"via BC during the pretraining phase."
		)
		self.alpha = alpha
		self.bc_data = bc_data
		self.bc_steps = bc_steps
		self.bc_coeff = bc_coeff

	def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
		# formalization step 2: parameterize the current policy as a neural network
		self.network = network_class(
			self.processor.observation_space, self.processor.action_space, **network_kwargs
		).to(self.device)

		# formalization step 3: obtain a frozen reference policy with the same
		# architecture as the current policy.
		# build_reference_network is in reference_policy.py -- swap that file to
		# replace the BC reference with BCO or another approach.
		# the reference_network is registered as an nn.Module via Algorithm.__setattr__,
		# which means it will be included in save/load and train/eval mode switching.
		self.reference_network = build_reference_network(
			self.processor.observation_space,
			self.processor.action_space,
			network_kwargs,
			self.device,
		)

	def setup_optimizers(self) -> None:
		# only the current policy (network) is optimized; reference_network is frozen
		params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
		groups = utils.create_optim_groups(params, self.optim_kwargs)
		self.optim["actor"] = self.optim_class(groups)

	def setup_schedulers(self, do_nothing=True):
		# suppress LR schedule during BC pretraining, activate after (mirrors CPL_KL)
		if do_nothing:
			for k in self.schedulers_class.keys():
				if k in self.optim:
					self.schedulers[k] = torch.optim.lr_scheduler.LambdaLR(
						self.optim[k], lr_lambda=lambda x: 1.0
					)
		else:
			self.schedulers = {}
			super().setup_schedulers()

	def _get_cpl_loss(self, batch):
		# requires "comparison" mode data: obs_1, action_1, obs_2, action_2, label
		assert "label" in batch, (
			"CPL_pAOT requires FeedbackBuffer with mode='comparison'. "
			"Got a batch without 'label' key."
		)

		# concatenate both segments so we can do a single encoder forward pass
		# shape after cat: (2*B, S, obs_dim) and (2*B, S, act_dim)
		obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
		action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)

		# formalization step 4 (current policy side):
		# compute log π_θ(a_t | s_t) for every timestep under the current policy
		obs_encoded = self.network.encoder(obs)
		dist = self.network.actor(obs_encoded)

		if isinstance(dist, torch.distributions.Distribution):
			lp = dist.log_prob(action)  # shape (2*B, S)
		else:
			# deterministic policy approximation (matches base CPL convention)
			assert dist.shape == action.shape
			lp = -torch.square(dist - action).sum(dim=-1)  # shape (2*B, S)

		# BC auxiliary loss -- keeps policy near behavior data
		if self.bc_data == "pos":
			lp1, lp2 = torch.chunk(lp, 2, dim=0)
			label = batch["label"]
			lp_pos = torch.cat(
				(lp1[label <= 0.5], lp2[label >= 0.5]), dim=0
			)
			bc_loss = (-lp_pos).mean()
		else:
			bc_loss = (-lp).mean()

		# formalization step 4 (reference policy side):
		# compute log π_ref(a_t | s_t) with no gradients
		# compute_reference_log_probs is in reference_policy.py
		with torch.no_grad():
			ref_lp = compute_reference_log_probs(self.reference_network, obs, action)
			# ref_lp shape: (2*B, S)

		# compute segment scores under π_θ:
		#   score(σ; π_θ) = Σ_t α log π_θ(a_t | s_t)
		# note: γ^t discounting omitted (γ=1 convention, matches base CPL)
		adv = self.alpha * lp  # shape (2*B, S)
		segment_adv = adv.sum(dim=-1)  # shape (2*B,)

		# compute segment scores under π_ref:
		#   score(σ; π_ref) = Σ_t α log π_ref(a_t | s_t)
		ref_adv = self.alpha * ref_lp  # shape (2*B, S)
		ref_segment_adv = ref_adv.sum(dim=-1)  # shape (2*B,)

		# split back into per-pair scores for both segments
		adv1, adv2 = torch.chunk(segment_adv, 2, dim=0)         # each (B,)
		ref_adv1, ref_adv2 = torch.chunk(ref_segment_adv, 2, dim=0)  # each (B,)

		# formalization step 4: identify preferred and rejected per pair
		#   label=1 → obs_2 is preferred, label=0 → obs_1 is preferred
		label = batch["label"].float()
		pref_mask = label > 0.5  # True where obs_2 is preferred

		# preferred and rejected segment scores under π_θ
		adv_pos = torch.where(pref_mask, adv2, adv1)
		adv_neg = torch.where(pref_mask, adv1, adv2)

		# preferred and rejected segment scores under π_ref
		ref_adv_pos = torch.where(pref_mask, ref_adv2, ref_adv1)
		ref_adv_neg = torch.where(pref_mask, ref_adv1, ref_adv2)

		# formalization step 4: compute per-pair margins
		#   u_θ^i = score(σ_i+; π_θ) - score(σ_i-; π_θ)
		#   v_ref^i = score(σ_i+; π_ref) - score(σ_i-; π_ref)
		u_theta = adv_pos - adv_neg    # policy margin per pair, shape (B,)
		v_ref = ref_adv_pos - ref_adv_neg  # reference margin per pair, shape (B,)

		# drop tied pairs (label=0.5) -- margins for ties are meaningless
		not_tied = label != 0.5
		u_theta = u_theta[not_tied]
		v_ref = v_ref[not_tied]

		# formalization steps 5 and 6: OT sort + CPL loss (see paot_loss above)
		cpl_loss, accuracy = paot_loss(u_theta, v_ref)

		return cpl_loss, bc_loss, accuracy

	def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
		cpl_loss, bc_loss, accuracy = self._get_cpl_loss(batch)

		# formalization step 7: backpropagate
		# phase 1 (step < bc_steps): pure BC pretraining
		#   both current policy and (implicitly) reference policy are being trained
		# phase 2 (step >= bc_steps): pAOT-CPL loss + optional BC regularization
		#   reference policy weights are frozen from the snapshot taken at transition
		if step < self.bc_steps:
			loss = bc_loss
			cpl_loss, accuracy = torch.tensor(0.0), torch.tensor(0.0)
		else:
			loss = cpl_loss + self.bc_coeff * bc_loss

		self.optim["actor"].zero_grad()
		loss.backward()
		self.optim["actor"].step()

		# at the BC→CPL transition:
		#   1. snapshot the BC-trained weights into the frozen reference network
		#      (formalization step 3: reference policy obtained via BC)
		#   2. reset optimizer and activate LR schedule for the CPL phase
		if step == self.bc_steps - 1:
			del self.optim["actor"]

			# snapshot_reference is in reference_policy.py
			# replace this call if using BCO or another reference approach
			snapshot_reference(self.reference_network, self.network)

			params = itertools.chain(
				self.network.actor.parameters(),
				self.network.encoder.parameters()
			)
			groups = utils.create_optim_groups(params, self.optim_kwargs)
			self.optim["actor"] = self.optim_class(groups)
			self.setup_schedulers(do_nothing=False)

		return dict(
			cpl_loss=cpl_loss.item(),
			bc_loss=bc_loss.item(),
			accuracy=accuracy.item()
		)

	def validation_step(self, batch: Any) -> Dict:
		with torch.no_grad():
			cpl_loss, bc_loss, accuracy = self._get_cpl_loss(batch)
		return dict(
			cpl_loss=cpl_loss.item(),
			bc_loss=bc_loss.item(),
			accuracy=accuracy.item()
		)

	def _get_train_action(self, obs: Any, step: int, total_steps: int):
		batch = dict(obs=obs)
		with torch.no_grad():
			action = self.predict(batch, is_batched=False, sample=True)
		return action
