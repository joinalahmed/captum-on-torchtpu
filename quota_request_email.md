Subject: Cloud TPU quota increase request for the TorchTPU Content Sprint, project gen-lang-client-0320701242 (v5e and v6e, us-east5)

Hello Cloud TPU team,

I am requesting a quota increase for Cloud TPU on project gen-lang-client-0320701242. I am participating in the TorchTPU Content Sprint (#TorchTPUSprint, submissions due 25 November 2026) with an entry on explainability as a performance lens for TorchTPU: Captum attribution across execution modes, with XProf. The single-chip work is done and has produced upstream fixes to TorchTPU and Captum; the remaining step is to validate those fixes and the write-up on multi-chip slices before the deadline.

Project and location

- Project ID: gen-lang-client-0320701242
- Primary zone: us-east5-b (fallback: us-central1-a)
- Billing is enabled and in good standing; the project has been running v6e-1 on-demand VMs in us-east5-b over the last two weeks.

Current limits and what blocked me

- TPUV5sLitepodServingPerProjectPerZoneForTPUAPI: limit 4 in us-east5-b. A v5litepod-8 create fails with "Quota limit 'TPUV5sLitepodServingPerProjectPerZoneForTPUAPI' has been exceeded. Limit: 4 in zone us-east5-b."
- tpu-v5s-litepod (on-demand): 16 chips at the project level, but no on-demand allocation in us-east5-a or us-east5-c ("Reservation not found" on create).
- tpu-v6e (on-demand): 16 chips at the project level.
- On-demand and spot v5litepod-4, and v6e-4 / v6e-8, have returned "Insufficient capacity" or "no more capacity in the zone" in us-east5-b and us-central1-a throughout 2026-11-01. I currently have two queued-resource requests waiting (v6e-4 in us-east5-b, v5litepod-4 in us-central1-a).

Requested changes

1. TPUV5sLitepodServingPerProjectPerZoneForTPUAPI in us-east5-b: raise from 4 to 8 chips, so a v5litepod-8 (the same configuration as the torch_tpu upstream CI runners) can be created.
2. tpu-v5s-litepod on-demand in us-east5-a and us-east5-c: 8 chips each, so I can fall back to a neighbouring zone when us-east5-b has no capacity.
3. tpu-v6e on-demand in us-east5-b: keep 16 chips, and if possible add 8 chips in us-east5-c for the same reason.
4. Preemptible/spot equivalents for the above at the same levels, so I can use spot capacity when on-demand is unavailable.

If any of these cannot be granted, guidance on which zones currently have v5e or v6e capacity for small slices (4 to 8 chips), or whether a short-term reservation is the recommended route, would be equally helpful. Given the 25 November deadline, anything that lands within the next two weeks makes the difference between a single-chip and a multi-chip submission.

Use case

The sprint entry validates TorchTPU changes on multi-chip hardware before submitting them upstream. Open pull requests and issues from this work: google-pytorch/torch_tpu#6102, #6105, #6106 (open PRs), #6103, #6111 to #6114 (issues with fixes on branches), and meta-pytorch/captum#1951, #1952. The single-chip results and the write-up are at https://github.com/joinalahmed/captum-on-torchtpu. The multi-chip runs are needed to cover the v5e test variants that upstream CI uses, to test the distributed code paths touched by two of the fixes, and to add a multi-chip attribution comparison to the sprint blog post and the TorchTPU docs case study that is part of the submission.

Expected usage

- Duration: intermittent until the sprint deadline on 25 November 2026, then occasional follow-up on the open PRs, in sessions of 4 to 10 hours. VMs are created per session and deleted afterwards.
- Peak concurrent usage: one 8-chip slice (v5e or v6e), occasionally one 4-chip slice alongside it for a few hours.
- Estimated monthly spend at these levels: USD 300 to 600.

Thank you,

Joinal Ahmed
GitHub: joinalahmed
GDE - Google Cloud & AI - TPU, vLLM
