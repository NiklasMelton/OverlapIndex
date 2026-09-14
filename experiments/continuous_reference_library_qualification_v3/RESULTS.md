# Reference-library qualification V3 results

## Identity

- Executed commit: `3a221542388500e6483221393418d0fcb784a74a`
- Protocol SHA-256: `807b9687a48d8025c9df54d742a06613e4bd89ecd04f7a934861c3e561961eb3`
- Manifest SHA-256: `c4c496986db626e5a6e1cd1ade7a6688f51c459b0400db1d96c1fee2126fe1cd`
- Raw-results SHA-256: `9dc3a96a7297048bc2f926298fa4d3fe7988e0a48391ff4266492c87c62f344d`
- Summary SHA-256: `46f56457d81b4c183dacc8fbec3f5bd29f1dac7ab5701c74490cc699367b5f41`
- Pre-execution verification: 31 focused tests passed
- Scope: 25 real candidate draws and 45 synthetic candidates; no
  OverlapIndex fits and no reserved backbones

## Decision

V3 **passed** every frozen gate and authorizes the eight-target historical
development screen with this exact head recipe:

- zero reported or actionable material parameter endpoints;
- all linear, quadratic, and smooth-periodic controls passed;
- both real CUB linear controls passed;
- the LSP nonlinear-head gain passed;
- no fit warnings occurred.

The clean synthetic candidate won all nine panels. Minimum intended-head R2
remained at least 0.9982; mean linear regret was 0.126 on quadratic and 0.213
on smooth-periodic mechanisms.

The real controls remained consistent. DINOv2-small won both CUB tasks under
the linear head and envelope with zero top-1 regret. ConvNeXt-tiny won LSP
under both rankings, while the best nonlinear head still added 0.184 R2.

## Reporting note

The immutable generated Markdown report inherited the text heading
"qualification V2" from its renderer. The terminal status, protocol, manifest,
summary, raw-result status, and artifact paths all correctly identify V3. This
is a cosmetic label defect only; it did not affect any fitted model,
calculation, or decision. The completed artifact was not mutated or rerun.

## Authorized next stage

The exact V3 head recipe may now be used without tuning on all eight cached
historical targets. That screen remains development-only because it has five
scout backbones and 200 evaluation rows per target. Its purpose is to classify
corrected targets as linear, nonlinear-challenge leads, or unresolved before
new datasets and reserved backbones are opened.
