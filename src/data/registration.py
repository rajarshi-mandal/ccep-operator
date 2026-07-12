"""Route A — real EPI->T1->MNI registration so Schaefer parcels land on the right anatomy.

Route B (raw affine overlay) was shown to be anatomically meaningless on ds005498: the evoked
response did not peak at the coil (parcel percentile 0.41, peak 85 mm away) and same-site
cross-subject topography coherence was ~0.01. The native EPI affines are NOT in register with
MNI, so overlaying the MNI atlas samples arbitrary tissue.

This module fixes that with classic intensity-based affine registration (dipy, pure Python —
no FSL/ANTs/fMRIPrep needed):

    atlas(MNI) --[T1<-MNI affine]--> atlas(T1) --[EPI<-T1 rigid]--> atlas(EPI grid)

We register the subject T1 to MNI (12-DOF affine) and the run's mean EPI to that T1 (6-DOF
rigid), then carry the Schaefer atlas backwards through both maps onto the native EPI grid via
nearest-neighbour. The 4-D BOLD is never resampled (cheap); only the small label volume moves.

The T1->MNI fit is the slow part, so ``register_subject`` caches it and reuses it across all of
a subject's runs (only the fast EPI->T1 rigid is per-geometry).
"""
from __future__ import annotations

import numpy as np
import nibabel as nib


def _affreg(quality: str = "fast"):
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    if quality == "fast":
        level_iters, sigmas, factors = [500, 100, 10], [3, 1, 0], [4, 2, 1]
    else:
        level_iters, sigmas, factors = [10000, 1000, 100], [3, 1, 0], [4, 2, 1]
    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    return AffineRegistration(metric=metric, level_iters=level_iters,
                              sigmas=sigmas, factors=factors, verbosity=0)


def _fit(static, static_aff, moving, moving_aff, kind: str, quality: str):
    """Run COM -> translation -> rigid (-> affine) and return the final AffineMap."""
    from dipy.align.imaffine import transform_centers_of_mass
    from dipy.align.transforms import (AffineTransform3D, RigidTransform3D,
                                       TranslationTransform3D)
    affreg = _affreg(quality)
    com = transform_centers_of_mass(static, static_aff, moving, moving_aff)
    tr = affreg.optimize(static, moving, TranslationTransform3D(), None,
                         static_aff, moving_aff, starting_affine=com.affine)
    rig = affreg.optimize(static, moving, RigidTransform3D(), None,
                          static_aff, moving_aff, starting_affine=tr.affine)
    if kind == "rigid":
        return rig
    return affreg.optimize(static, moving, AffineTransform3D(), None,
                           static_aff, moving_aff, starting_affine=rig.affine)


def schaefer_on_template_grid(atlas_img, template_img):
    """The atlas already lives in MNI; ensure the registration target (template) shares the
    atlas grid so we can carry atlas labels through the same AffineMap. Resample the MNI152
    template onto the Schaefer grid (both true MNI -> valid affine resample)."""
    from nilearn.image import resample_to_img
    t = resample_to_img(template_img, atlas_img, interpolation="continuous",
                        force_resample=True, copy_header=True)
    return np.asarray(t.dataobj, dtype=np.float32)


class SubjectRegistration:
    """Caches the subject's T1->MNI fit; maps the Schaefer atlas onto any of the subject's
    EPI grids on demand."""

    def __init__(self, t1_path, atlas_img, template_img, quality: str = "fast"):
        self.atlas_img = atlas_img
        self.atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
        self.atlas_aff = atlas_img.affine
        self.quality = quality
        # static = MNI template on the Schaefer grid (so atlas shares this grid)
        self.tmpl_on_atlas = schaefer_on_template_grid(atlas_img, template_img)
        t1 = nib.load(str(t1_path))
        self.t1_data = np.asarray(t1.dataobj, dtype=np.float32)
        self.t1_aff = t1.affine
        # T1 -> MNI (atlas grid)
        self.map_t1 = _fit(self.tmpl_on_atlas, self.atlas_aff, self.t1_data, self.t1_aff,
                           kind="affine", quality=quality)
        # atlas carried into T1 space once (reused for every EPI of this subject)
        self.atlas_in_t1 = self.map_t1.transform_inverse(
            self.atlas_data, interpolation="nearest").astype(np.int32)

    def atlas_in_epi(self, epi_img) -> np.ndarray:
        """Return the Schaefer label volume on this EPI's grid (rigid EPI->T1)."""
        mean_epi = np.asarray(epi_img.dataobj, dtype=np.float32)
        if mean_epi.ndim == 4:
            mean_epi = mean_epi.mean(-1)
        map_epi = _fit(self.t1_data, self.t1_aff, mean_epi, epi_img.affine,
                       kind="rigid", quality=self.quality)
        return map_epi.transform_inverse(self.atlas_in_t1,
                                         interpolation="nearest").astype(np.int32)
