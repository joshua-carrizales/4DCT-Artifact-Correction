from correct_nifti_artifact_image import *
from interp_module_3D import *
from Networks_R3U import *
from preprocess_functions import *

### Instructions for Use ###
# Reference phases must be adjacent to the artifact phase.
# Ex: Ref 1 = 40IN, Artifact Phase = 60IN, Ref 2 = 80IN
# Can toggle to add an artificial interpolation artifact or phase shadow artifact.
# If you have data with artifacts in them already, set the toggle to False.

correct_NIFTI_image('/Insert/path/to/model/here',
                    '/Insert/path/to/4DCT/data/here',
                    '/Insert/path/to/save/corrected/images/here',
                    'Insert Reference Phase 1 Here', 'Insert Artifact Phase Here', 'Insert Reference Phase 2 Here',
                    add_artificial_interpolation=True,
                    add_artificial_phase_shadow=False
                    )