
# cpl_paot.py
#
# CPL with Paired Alignment via Optimal Transport (pAOT).
# Implements the algorithm described in aot_cpl.tex, section:
#   "CPL with pAOT"
#
# Training has three sequential phases:
#
#   Phase 1 -- Reference BC (steps 0 to ref_bc_steps)
#     π_ref (reference_network) is trained with behavior cloning.
#     π_θ (network) is completely untouched -- it stays at random init.
#     Corresponds to formalization step 3: "obtain reference policy via BC".
#
#   Phase 2 -- Optional policy warmup (steps ref_bc_steps to ref_bc_steps + theta_bc_steps)
#     π_ref is frozen (weights locked after phase 1).
#     π_θ is optionally warmed up with behavior cloning.
#     Set theta_bc_steps=0 to skip this phase and start pAOT from a fully
#     fresh π_θ, which stays true to the formalization.
#
#   Phase 3 -- pAOT contrastive training (steps ref_bc_steps + theta_bc_steps onwards)
#     π_θ is trained with the pAOT loss using the frozen π_ref as a reference.
#     Corresponds to formalization steps 4-7.
#
# Key difference from uAOT:
#   uAOT sorts raw preferred/rejected segment scores independently.
#   pAOT operates on per-pair MARGINS (preferred_score - rejected_score) under
#   both π_θ and π_ref. Both margin sets are sorted independently via OT, then
#   the loss pushes π_θ margins to exceed π_ref margins.
#
# Data mode requirement:
#   FeedbackBuffer must use mode="comparison" so that each batch contains
#   (obs_1, action_1, obs_2, action_2, label). Per-pair margins require
#   knowing which segment in each pair is preferred vs. rejected.
#
# Discount factor note:
#   The formalization defines score_π(σ) = Σ_t γ^t α log π(a_t|s_t).
#   Baseline CPL does NOT apply γ^t (equivalent to γ=1). We match that
#   convention here for consistency.

import itertools
from typing import Any, Dict, Type

import torch

from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm
from .reference_policy import (
	build_reference_network,
	compute_reference_log_probs,
	freeze_reference,
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
# Interpretation: the loss pushes π_θ to achieve a higher within-pair
# margin than π_ref achieves, after OT re-matching.
def paot_loss(u_theta, v_ref):
	# step 5: sort policy margins and reference margins independently
	u_sorted = torch.sort(u_theta).values  # u_θ^(1) ≤ ... ≤ u_θ^(n)
	v_sorted = torch.sort(v_ref).values    # v_ref^(1) ≤ ... ≤ v_ref^(n)

	# step 6: compute loss on OT-matched margin pairs
	# logit > 0 when the policy margin exceeds the reference margin
	logit = u_sorted - v_sorted

	# numerically stable BCE: -log sigmoid(logit) = log(1 + exp(-logit))
	max_val = torch.clamp(-logit, min=0)
	loss = (torch.log(torch.exp(-max_val) + torch.exp(-logit - max_val)) + max_val).mean()

	# accuracy: fraction of OT-matched pairs where policy margin > reference margin
	with torch.no_grad():
		accuracy = (u_sorted > v_sorted).float().mean()

	return loss, accuracy


class CPL_pAOT(OffPolicyAlgorithm):
	def __init__(
		self,
		*args,
		alpha: float = 1.0,
		ref_bc_steps: int = 200000,  # phase 1: how long to BC-train π_ref
		theta_bc_steps: int = 0,     # phase 2: how long to BC-warm π_θ (0 = skip)
		bc_coeff: float = 0.0,       # optional BC regularization during pAOT phase
		bc_data: str = "all",        # which segments to use for BC ("all" or "pos")
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		assert "encoder" in self.network.CONTAINERS
		assert "actor" in self.network.CONTAINERS
		assert ref_bc_steps > 0, "ref_bc_steps must be > 0 to BC-train the reference policy"
		self.alpha = alpha
		self.ref_bc_steps = ref_bc_steps
		self.theta_bc_steps = theta_bc_steps
		self.bc_coeff = bc_coeff
		self.bc_data = bc_data
		# the step at which pAOT contrastive training begins
		self._paot_start = ref_bc_steps + theta_bc_steps

	def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
		# π_θ (formalization step 2): fresh policy network, trained contrastively in phase 3
		# initialized randomly and never touched during phase 1 (reference BC)
		self.network = network_class(
			self.processor.observation_space, self.processor.action_space, **network_kwargs
		).to(self.device)

		# π_ref (formalization step 3): reference policy network, BC-trained in phase 1
		# gradients are enabled here so phase 1 can train it directly
		# freeze_reference() is called at the end of phase 1 to lock the weights
		self.reference_network = build_reference_network(
			self.processor.observation_space,
			self.processor.action_space,
			network_kwargs,
			self.device,
		)

	def setup_optimizers(self) -> None:
		# start with optimizer over π_ref for phase 1 BC training
		# this will be replaced at the end of phase 1 with an optimizer over π_θ
		params = itertools.chain(
			self.reference_network.actor.parameters(),
			self.reference_network.encoder.parameters()
		)
		groups = utils.create_optim_groups(params, self.optim_kwargs)
		self.optim["actor"] = self.optim_class(groups)

	def setup_schedulers(self, do_nothing=True):
		# suppress LR schedule during BC phases, activate when pAOT begins
		if do_nothing:
			for k in self.schedulers_class.keys():
				if k in self.optim:
					self.schedulers[k] = torch.optim.lr_scheduler.LambdaLR(
						self.optim[k], lr_lambda=lambda x: 1.0
					)
		else:
			self.schedulers = {}
			super().setup_schedulers()

	def _get_bc_loss(self, batch, network):
		# compute BC loss (negative log-likelihood) on a given network
		# used for both phase 1 (network=reference_network) and phase 2 (network=self.network)
		obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
		action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)

		dist = network.actor(network.encoder(obs))
		if isinstance(dist, torch.distributions.Distribution):
			lp = dist.log_prob(action)
		else:
			assert dist.shape == action.shape
			lp = -torch.square(dist - action).sum(dim=-1)

		if self.bc_data == "pos":
			lp1, lp2 = torch.chunk(lp, 2, dim=0)
			label = batch["label"]
			lp_pos = torch.cat((lp1[label <= 0.5], lp2[label >= 0.5]), dim=0)
			return (-lp_pos).mean()
		else:
			return (-lp).mean()

	def _get_cpl_loss(self, batch):
		# called during phase 3 only
		# requires "comparison" mode data: obs_1, action_1, obs_2, action_2, label
		assert "label" in batch, (
			"CPL_pAOT requires FeedbackBuffer with mode='comparison'. "
			"Got a batch without 'label' key."
		)

		obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
		action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)

		# formalization step 4 (π_θ side):
		# compute log π_θ(a_t | s_t) for every timestep
		obs_encoded = self.network.encoder(obs)
		dist = self.network.actor(obs_encoded)

		if isinstance(dist, torch.distributions.Distribution):
			lp = dist.log_prob(action)  # shape (2*B, S)
		else:
			assert dist.shape == action.shape
			lp = -torch.square(dist - action).sum(dim=-1)

		bc_loss = self._get_bc_loss(batch, self.network)

		# formalization step 4 (π_ref side):
		# compute log π_ref(a_t | s_t) with no gradients (π_ref is frozen)
		with torch.no_grad():
			ref_lp = compute_reference_log_probs(self.reference_network, obs, action)
			# ref_lp shape: (2*B, S)

		# compute segment scores under π_θ:
		#   score(σ; π_θ) = Σ_t α log π_θ(a_t | s_t)
		# note: γ^t discounting omitted (γ=1, matches base CPL convention)
		adv = self.alpha * lp
		segment_adv = adv.sum(dim=-1)  # shape (2*B,)

		# compute segment scores under π_ref:
		#   score(σ; π_ref) = Σ_t α log π_ref(a_t | s_t)
		ref_adv = self.alpha * ref_lp
		ref_segment_adv = ref_adv.sum(dim=-1)  # shape (2*B,)

		# split into per-pair scores
		adv1, adv2 = torch.chunk(segment_adv, 2, dim=0)
		ref_adv1, ref_adv2 = torch.chunk(ref_segment_adv, 2, dim=0)

		# formalization step 4: identify preferred and rejected per pair
		#   label=1 → obs_2 is preferred, label=0 → obs_1 is preferred
		label = batch["label"].float()
		pref_mask = label > 0.5

		adv_pos = torch.where(pref_mask, adv2, adv1)
		adv_neg = torch.where(pref_mask, adv1, adv2)
		ref_adv_pos = torch.where(pref_mask, ref_adv2, ref_adv1)
		ref_adv_neg = torch.where(pref_mask, ref_adv1, ref_adv2)

		# formalization step 4: per-pair margins
		#   u_θ^i = score(σ_i+; π_θ) - score(σ_i-; π_θ)
		#   v_ref^i = score(σ_i+; π_ref) - score(σ_i-; π_ref)
		u_theta = adv_pos - adv_neg
		v_ref = ref_adv_pos - ref_adv_neg

		# drop tied pairs (label=0.5) -- margins for ties are meaningless
		not_tied = label != 0.5
		u_theta = u_theta[not_tied]
		v_ref = v_ref[not_tied]

		# formalization steps 5 and 6: OT sort + CPL loss (see paot_loss above)
		cpl_loss, accuracy = paot_loss(u_theta, v_ref)

		return cpl_loss, bc_loss, accuracy

	def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
		# -- phase 1: BC training of π_ref (formalization step 3) --
		# π_θ is completely untouched during this phase
		if step < self.ref_bc_steps:
			bc_loss = self._get_bc_loss(batch, self.reference_network)
			self.optim["actor"].zero_grad()
			bc_loss.backward()
			self.optim["actor"].step()

			# end of phase 1: freeze π_ref, switch optimizer to π_θ
			if step == self.ref_bc_steps - 1:
				freeze_reference(self.reference_network)
				del self.optim["actor"]
				params = itertools.chain(
					self.network.actor.parameters(),
					self.network.encoder.parameters()
				)
				groups = utils.create_optim_groups(params, self.optim_kwargs)
				self.optim["actor"] = self.optim_class(groups)
				# activate LR schedule now if there is no theta BC warmup phase
				if self.theta_bc_steps == 0:
					self.setup_schedulers(do_nothing=False)

			return dict(
				cpl_loss=0.0,
				bc_loss=bc_loss.item(),
				accuracy=0.0
			)

		# -- phase 2: optional BC warmup of π_θ --
		# π_ref is frozen, π_θ trained with BC before contrastive loss kicks in
		elif step < self._paot_start:
			bc_loss = self._get_bc_loss(batch, self.network)
			self.optim["actor"].zero_grad()
			bc_loss.backward()
			self.optim["actor"].step()

			# end of phase 2: reset optimizer and activate LR schedule for pAOT
			if step == self._paot_start - 1:
				del self.optim["actor"]
				params = itertools.chain(
					self.network.actor.parameters(),
					self.network.encoder.parameters()
				)
				groups = utils.create_optim_groups(params, self.optim_kwargs)
				self.optim["actor"] = self.optim_class(groups)
				self.setup_schedulers(do_nothing=False)

			return dict(
				cpl_loss=0.0,
				bc_loss=bc_loss.item(),
				accuracy=0.0
			)

		# -- phase 3: pAOT contrastive training of π_θ (formalization steps 4-7) --
		# π_ref is frozen throughout, π_θ is trained with the pAOT loss
		else:
			cpl_loss, bc_loss, accuracy = self._get_cpl_loss(batch)
			loss = cpl_loss + self.bc_coeff * bc_loss
			self.optim["actor"].zero_grad()
			loss.backward()
			self.optim["actor"].step()

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
