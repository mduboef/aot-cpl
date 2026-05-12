
# cpl_uaot.py
#
# CPL with Unpaired Alignment via Optimal Transport (uAOT).
# Implements the algorithm described in aot_cpl.tex, section:
#   "CPL with uAOT and No Reference Policy"
# Extended to use a reference policy for log-ratio scoring.
#
# Training has three sequential phases:
#
#   Phase 1 -- Reference BC (steps 0 to ref_bc_steps)
#     π_ref (reference_network) is trained with behavior cloning.
#     π_θ (network) is completely untouched -- it stays at random init.
#     Produces the reference policy used to normalize segment scores.
#
#   Phase 2 -- Optional policy warmup (steps ref_bc_steps to ref_bc_steps + bc_steps)
#     π_ref is frozen (weights locked after phase 1).
#     π_θ is optionally warmed up with behavior cloning.
#     Set bc_steps=0 to skip this phase and start uAOT from a fresh π_θ.
#
#   Phase 3 -- uAOT contrastive training (steps ref_bc_steps + bc_steps onwards)
#     π_θ is trained with the uAOT loss using frozen π_ref for log-ratio scores.
#
# Key difference from baseline CPL (cpl.py):
#   Standard CPL compares each (σ+, σ-) pair directly.
#   uAOT separates all preferred and rejected segments across the batch,
#   sorts each group independently by score, then matches the i-th ranked
#   preferred segment with the i-th ranked rejected segment (1-D optimal
#   transport via the northwest corner method). The CPL loss is then applied
#   to these OT-matched pairs rather than the original preference pairs.
#
# Key difference from pAOT (cpl_paot.py):
#   pAOT operates on per-pair MARGINS (preferred_score - rejected_score) and
#   sorts those margins, training π_θ margins to exceed π_ref margins.
#   uAOT operates on individual segment scores -- preferred and rejected sets
#   are sorted independently to achieve stochastic dominance of the preferred
#   score distribution over the rejected score distribution.
#
# Why log-ratio scores are required:
#   Absolute log-prob scores (α log π_θ(a|s)) are not comparable across
#   preference pairs -- they reflect both trajectory quality AND ease of
#   imitation. After BC warmup, some rejected segments (simple, predictable
#   actions) can outscore preferred segments (complex, specific movements),
#   causing OT to create semantically inverted cross-pair comparisons.
#   Log-ratio scores α(log π_θ - log π_ref) normalize out ease-of-imitation:
#   all scores start near 0 and shift only as π_θ improves beyond π_ref,
#   making cross-pair comparisons meaningful.
#   See formalization note at the end of the uAOT section in aot_cpl.tex.
#
# Discount factor note:
#   The formalization defines score_π(σ) = Σ_t γ^t α log π(a_t|s_t).
#   Baseline CPL does NOT apply γ^t (equivalent to γ=1). We match that
#   convention here for consistency. The FeedbackBuffer provides a
#   "discount" key in the batch but it is intentionally not used.

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


# uaot_loss
# Formalization reference: Steps 5 and 6 of "CPL with uAOT and No Reference Policy"
#
# Inputs:
#   u  -- preferred segment log-ratio scores for the batch, shape (n,)
#          u_θ^i = Σ_t α (log π_θ(a_t^{i,+} | s_t^{i,+}) - log π_ref(...))
#   v  -- rejected segment log-ratio scores for the batch, shape (n,)
#          v_θ^i = Σ_t α (log π_θ(a_t^{i,-} | s_t^{i,-}) - log π_ref(...))
#
# Step 5 (tex): Sort u and v independently from lowest to highest.
#   u_θ^(1) ≤ u_θ^(2) ≤ ... ≤ u_θ^(n)
#   v_θ^(1) ≤ v_θ^(2) ≤ ... ≤ v_θ^(n)
#   The i-th ranked preferred score is matched with the i-th ranked rejected score.
#
# Step 6 (tex): Compute the CPL loss on the OT-matched pairs.
#   L(θ) = (1/n) Σ_i -log [ exp(u_θ^(i)) / (exp(u_θ^(i)) + exp(v_θ^(i))) ]
#
# Implementation uses the numerically stable log-sum-exp trick to avoid overflow.
def uaot_loss(u, v):
	# step 5: sort preferred and rejected scores independently
	u_sorted = torch.sort(u).values  # u_θ^(1) ≤ ... ≤ u_θ^(n)
	v_sorted = torch.sort(v).values  # v_θ^(1) ≤ ... ≤ v_θ^(n)

	# step 6: compute loss on OT-matched pairs
	# logit = u_sorted - v_sorted (the log-odds favoring the preferred segment)
	logit = u_sorted - v_sorted

	# numerically stable BCE: -log sigmoid(logit) = log(1 + exp(-logit))
	# implemented via log-sum-exp: log(exp(0) + exp(-logit)) = log(1 + exp(-logit))
	max_val = torch.clamp(-logit, min=0)
	loss = (torch.log(torch.exp(-max_val) + torch.exp(-logit - max_val)) + max_val).mean()

	# accuracy: fraction of OT-matched pairs where preferred score > rejected score
	with torch.no_grad():
		accuracy = (u_sorted > v_sorted).float().mean()

	return loss, accuracy


class CPL_uAOT(OffPolicyAlgorithm):
	def __init__(
		self,
		*args,
		alpha: float = 1.0,
		bc_coeff: float = 0.0,
		bc_data: str = "all",
		ref_bc_steps: int = 200000,  # phase 1: how long to BC-train π_ref
		bc_steps: int = 0,           # phase 2: how long to BC-warm π_θ (0 = skip)
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		assert "encoder" in self.network.CONTAINERS
		assert "actor" in self.network.CONTAINERS
		assert ref_bc_steps > 0, "ref_bc_steps must be > 0 to BC-train the reference policy"
		self.alpha = alpha
		self.bc_data = bc_data
		self.bc_steps = bc_steps
		self.bc_coeff = bc_coeff
		self.ref_bc_steps = ref_bc_steps
		# the step at which uAOT contrastive training begins
		self._uaot_start = ref_bc_steps + bc_steps

	def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
		# π_θ: fresh policy network, trained contrastively in phase 3
		# initialized randomly and never touched during phase 1 (reference BC)
		self.network = network_class(
			self.processor.observation_space, self.processor.action_space, **network_kwargs
		).to(self.device)

		# π_ref: reference policy network, BC-trained in phase 1
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
		# suppress LR schedule during BC phases, activate when uAOT begins
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
		# requires "comparison" mode data: obs_1, action_1, obs_2, action_2, label
		assert "label" in batch, (
			"CPL_uAOT requires FeedbackBuffer with mode='comparison'. "
			"Got a batch without 'label' key."
		)

		# concatenate both segments so we can do a single encoder forward pass
		# shape after cat: (2*B, S, obs_dim) and (2*B, S, act_dim)
		obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
		action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)

		# compute log π_θ(a_t | s_t) for every timestep
		obs_encoded = self.network.encoder(obs)
		dist = self.network.actor(obs_encoded)

		if isinstance(dist, torch.distributions.Distribution):
			lp = dist.log_prob(action)  # shape (2*B, S)
		else:
			# deterministic policy approximation (matches base CPL convention)
			assert dist.shape == action.shape
			lp = -torch.square(dist - action).sum(dim=-1)  # shape (2*B, S)

		bc_loss = self._get_bc_loss(batch, self.network)

		# compute log π_ref(a_t | s_t) with no gradients (π_ref is frozen)
		with torch.no_grad():
			ref_lp = compute_reference_log_probs(self.reference_network, obs, action)
			# ref_lp shape: (2*B, S)

		# formalization note (uAOT section): log-ratio scores normalize out ease-of-imitation
		#   u_θ^i = Σ_t α (log π_θ(a_t^{i,+}|s_t^{i,+}) - log π_ref(a_t^{i,+}|s_t^{i,+}))
		#   v_θ^i = Σ_t α (log π_θ(a_t^{i,-}|s_t^{i,-}) - log π_ref(a_t^{i,-}|s_t^{i,-}))
		# note: γ^t discounting is omitted (γ=1 convention, matches base CPL)
		adv = self.alpha * (lp - ref_lp)  # shape (2*B, S)
		segment_adv = adv.sum(dim=-1)     # shape (2*B,) -- sum over timesteps

		# split back into per-segment scores for the two sides of each pair
		adv1, adv2 = torch.chunk(segment_adv, 2, dim=0)  # each (B,)

		# formalization step 4: split into preferred and rejected sets
		#   label=1 means obs_2 is preferred, label=0 means obs_1 is preferred
		label = batch["label"].float()
		pref_mask = label > 0.5  # True where obs_2 is preferred

		# u = score of the preferred segment for each pair
		# v = score of the rejected segment for each pair
		u = torch.where(pref_mask, adv2, adv1)
		v = torch.where(pref_mask, adv1, adv2)

		# drop tied pairs (label=0.5) -- OT matching requires a clear preference
		not_tied = label != 0.5
		u = u[not_tied]
		v = v[not_tied]

		# formalization steps 5 and 6: OT sort + CPL loss (see uaot_loss above)
		cpl_loss, accuracy = uaot_loss(u, v)

		return cpl_loss, bc_loss, accuracy

	def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
		# -- phase 1: BC training of π_ref --
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
				if self.bc_steps == 0:
					self.setup_schedulers(do_nothing=False)

			return dict(cpl_loss=0.0, bc_loss=bc_loss.item(), accuracy=0.0)

		# -- phase 2: optional BC warmup of π_θ --
		# π_ref is frozen, π_θ trained with BC before contrastive loss kicks in
		elif step < self._uaot_start:
			bc_loss = self._get_bc_loss(batch, self.network)
			self.optim["actor"].zero_grad()
			bc_loss.backward()
			self.optim["actor"].step()

			# end of phase 2: reset optimizer and activate LR schedule for uAOT
			if step == self._uaot_start - 1:
				del self.optim["actor"]
				params = itertools.chain(
					self.network.actor.parameters(),
					self.network.encoder.parameters()
				)
				groups = utils.create_optim_groups(params, self.optim_kwargs)
				self.optim["actor"] = self.optim_class(groups)
				self.setup_schedulers(do_nothing=False)

			return dict(cpl_loss=0.0, bc_loss=bc_loss.item(), accuracy=0.0)

		# -- phase 3: uAOT contrastive training of π_θ --
		# π_ref is frozen throughout, π_θ is trained with log-ratio uAOT loss
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
