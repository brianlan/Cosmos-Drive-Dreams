These models are shown on the README.md, which should I donwload?
---

• For style transfer with your HDMap controls:

  - Single-view generation (step 4): Use Cosmos-Transfer1-7B-Sample-AV. That’s the model you need to
    download and point --checkpoint_dir to.
  - Multiview expansion (step 5, optional): Use Cosmos-7B-Single2Multiview-Sample-AV.

  You do not need the base_model, multiview base, annotate, or LiDAR-GEN for this pipeline.

  Suggested layout after download:

  checkpoints/
    cosmos_transfer1_7b_sample_av/    # weights from Cosmos-Transfer1-7B-Sample-AV
    cosmos_single2multiview_7b/       # weights from Cosmos-7B-Single2Multiview-Sample-AV (only if
  you’ll do multiview)

  Once Cosmos-Transfer1-7B-Sample-AV is in place and cosmos-transfer1 is on PYTHONPATH, rerun the
  single-view command (it will now find the package and the checkpoint). If you want, tell me where
  you’ll place the checkpoint folders and I can wire the exact paths into the command.


› Does Multiview expansion require the original multiview images (like back, left_back, etc.)? Or it
  only depend on the front view?


Does Multiview expansion require the original multiview images (like back, left_back, etc.)? Or it only depend on the front view?
---

• Multiview expansion does not need the original RGB multiview shots. The pipeline is:

  - Step 4: generate a single front-view RGB with Cosmos-Transfer1-7B-Sample-AV.
  - Step 5: Cosmos-Single2Multiview takes that front-view output plus your multiview control signals
    (HDMap per camera) to synthesize the other views.

  So you only need:

  - The front-view single_view output.
  - The rendered control videos for all cameras (HDMap/world-scenario). No original multiview RGB is
    required.
