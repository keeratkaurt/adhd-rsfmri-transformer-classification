# Data

Place the derived functional-connectivity data at `data/connectivity.npz`.

The archive must contain:

- `X`: shape `(n_participants, 116, 116)`, one AAL-116 Pearson-correlation matrix per participant
- `y`: shape `(n_participants,)`, with `0 = control` and `1 = ADHD`

The original UCLA Consortium for Neuropsychiatric Phenomics imaging data are not redistributed here. Follow the dataset's access and citation requirements, and do not commit participant-level derivatives that you are not authorized to share.
