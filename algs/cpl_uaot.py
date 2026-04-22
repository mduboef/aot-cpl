
# cpl_uaot.py
#
# CPL with Unpaired Alignment via Optimal Transport (uAOT).
# Implements the algorithm described in aot_cpl.tex, section:
#   "CPL with uAOT and No Reference Policy"
#
# Key difference from baseline CPL (cpl.py):
#   Standard CPL compares each (σ+, σ-) pair directly.
#   uAOT instead separates all preferred and rejected segments across the batch,
#   sorts each group independently by score, then matches the i-th ranked
#   preferred segment with the i-th ranked rejected segment (1-D optimal
#   transport via the northwest corner method). The CPL loss is then applied
#   to these OT-matched pairs rather than the original preference pairs.
#
# This allows the model to learn from cross-pair comparisons: a highly-ranked
# preferred segment is compared against a highly-ranked rejected one, which
# may provide a stronger training signal than within-pair comparisons alone.
#
# Data mode requirement:
#   FeedbackBuffer must use mode="comparison" so that each batch contains
#   (obs_1, action_1, obs_2, action_2, label). The label indicates which
#   segment is preferred (label=1 → obs_2 is preferred, label=0 → obs_1).
#   This is different from baseline CPL configs which use mode="rank".
#
# Discount factor note:
#   The formalization defines score_π(σ) = Σ_t γ^t α log π(a_t|s_t).
#   Baseline CPL does NOT apply γ^t (equivalent to γ=1). We match that
#   convention here for consistency. The FeedbackBuffer provides a
#   "discount" key in the batch but it is intentionally not used.

import itertools
from typing import Any, Dict

import torch

from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm


# uaot_loss
# Formalization reference: Steps 5 and 6 of "CPL with uAOT and No Reference Policy"
#
# Inputs:
#   u  -- preferred segment scores for the batch, shape (n,)
#          u_θ^i = Σ_t α log π_θ(a_t^{i,+} | s_t^{i,+})  (tex eq. after Step 3)
#   v  -- rejected segment scores for the batch, shape (n,)
#          v_θ^i = Σ_t α log π_θ(a_t^{i,-} | s_t^{i,-})  (tex eq. after Step 3)
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
		bc_steps: int = 0,
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		assert "encoder" in self.network.CONTAINERS
		assert "actor" in self.network.CONTAINERS
		self.alpha = alpha
		self.bc_data = bc_data
		self.bc_steps = bc_steps
		self.bc_coeff = bc_coeff
		# note: no contrastive_bias here -- the uAOT loss is unbiased BCE
		# (the tex formalization does not include the bias term from base CPL)

	def setup_optimizers(self) -> None:
		params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
		groups = utils.create_optim_groups(params, self.optim_kwargs)
		self.optim["actor"] = self.optim_class(groups)

	def setup_schedulers(self, do_nothing=True):
		# mirror CPL: suppress LR schedule during BC pretraining, activate after
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
			"CPL_uAOT requires FeedbackBuffer with mode='comparison'. "
			"Got a batch without 'label' key."
		)

		# concatenate both segments so we can do a single encoder forward pass
		# shape after cat: (2*B, S, obs_dim) and (2*B, S, act_dim)
		obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
		action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)

		# formalization step 3: compute log π_θ(a_t | s_t) for every timestep
		# obs is encoded first, then passed through actor to get a distribution
		obs_encoded = self.network.encoder(obs)
		dist = self.network.actor(obs_encoded)

		if isinstance(dist, torch.distributions.Distribution):
			# stochastic policy: true NLL
			lp = dist.log_prob(action)  # shape (2*B, S)
		else:
			# deterministic policy approximation (matches base CPL convention)
			assert dist.shape == action.shape
			lp = -torch.square(dist - action).sum(dim=-1)  # shape (2*B, S)

		# BC auxiliary loss (same structure as base CPL)
		# keeps policy close to behavior data during and after pretraining
		if self.bc_data == "pos":
			lp1, lp2 = torch.chunk(lp, 2, dim=0)
			label = batch["label"]
			lp_pos = torch.cat(
				(lp1[label <= 0.5], lp2[label >= 0.5]), dim=0
			)
			bc_loss = (-lp_pos).mean()
		else:
			bc_loss = (-lp).mean()

		# formalization step 3: compute segment scores for each segment
		#   u_θ^i = Σ_t α log π_θ(a_t^{i,+} | s_t^{i,+})
		#   v_θ^i = Σ_t α log π_θ(a_t^{i,-} | s_t^{i,-})
		# note: γ^t discounting is omitted (γ=1 convention, matches base CPL)
		adv = self.alpha * lp  # shape (2*B, S)
		segment_adv = adv.sum(dim=-1)  # shape (2*B,) -- sum over timesteps

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
		# ties would introduce random preferred/rejected assignments
		not_tied = label != 0.5
		u = u[not_tied]
		v = v[not_tied]

		# formalization steps 5 and 6: OT sort + CPL loss (see uaot_loss above)
		cpl_loss, accuracy = uaot_loss(u, v)

		return cpl_loss, bc_loss, accuracy

	def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
		cpl_loss, bc_loss, accuracy = self._get_cpl_loss(batch)

		# formalization step 7: backpropagate
		# phase 1 (step < bc_steps): pure BC pretraining
		# phase 2 (step >= bc_steps): uAOT-CPL loss + optional BC regularization
		if step < self.bc_steps:
			loss = bc_loss
			cpl_loss, accuracy = torch.tensor(0.0), torch.tensor(0.0)
		else:
			loss = cpl_loss + self.bc_coeff * bc_loss

		self.optim["actor"].zero_grad()
		loss.backward()
		self.optim["actor"].step()

		# at the BC→CPL transition: reset optimizer and activate LR schedule
		# (mirrors base CPL to ensure a clean start for the CPL phase)
		if step == self.bc_steps - 1:
			del self.optim["actor"]
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
