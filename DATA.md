# Data provenance and usage terms

This project uses the **MovieLens 100K** stable benchmark from the GroupLens
Research Project at the University of Minnesota. The dataset is fetched directly
from GroupLens at setup time; no MovieLens ratings, metadata, archive, or extracted
files are distributed in this repository.

## Exact source

- Dataset: MovieLens 100K, released April 1998
- Dataset page: <https://grouplens.org/datasets/movielens/100k/>
- Archive: <https://files.grouplens.org/datasets/movielens/ml-100k.zip>
- Authoritative README and terms:
  <https://files.grouplens.org/datasets/movielens/ml-100k-README.txt>
- GroupLens checksum file:
  <https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5>

The pinned archive contains 100,000 integer ratings on a 1--5 scale from 943
numbered user records for 1,682 movies. Each included user rated at least 20
movies. GroupLens reports that the data were collected through MovieLens from
September 19, 1997, through April 22, 1998. The project does not claim that the
records are anonymous or risk-free merely because names are absent.

The downloader verifies the exact official object before extraction:

```text
Size:    4,924,029 bytes
MD5:     0e33842e24a9c977be4e0107933c0723
SHA-256: 50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229
```

The MD5 value is published by GroupLens. SHA-256 is an additional repository pin;
both digests and the exact byte length must match. The same SHA-256 and byte length
are independently recorded in the
[TensorFlow Datasets MovieLens checksum manifest](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/movielens.txt).

## License boundary

MovieLens data are **not** covered by this repository's MIT License. GroupLens's
accompanying README says the dataset may be used for research purposes subject to
conditions that include:

- do not state or imply endorsement by the University of Minnesota or GroupLens;
- acknowledge the dataset in publications resulting from its use;
- do not redistribute the data without separate permission; and
- do not use the data for commercial or revenue-bearing purposes without first
  obtaining permission from a GroupLens faculty member.

Read the authoritative README linked above for its complete and controlling
language. This summary is not legal advice, does not grant additional rights, and
should not be treated as a substitute for checking the source terms before use.

Passing `--accept-license` signals only your explicit decision to proceed after
reading those terms. It does not change them or transfer rights to you. If your use
is commercial, revenue-bearing, or involves redistribution, obtain the necessary
permission directly from GroupLens first.

The repository's ownership and software-license boundary is summarized in
[NOTICE.md](NOTICE.md). It does not modify or supersede the GroupLens terms.

## Reproducible acquisition

Python 3.10 or later is sufficient; the downloader uses only the standard library.

```bash
python download_data.py --accept-license
```

The script:

1. requires explicit acceptance before making a network request;
2. permits HTTPS only, including redirects;
3. downloads to a temporary file in `data/raw/`;
4. validates the archive byte length, SHA-256, and GroupLens-published MD5, then
   checks pinned SHA-256 digests for `README`, `u.data`, `u.genre`, `u.info`, and
   `u.item` after extraction;
5. rejects unsafe paths, links, encryption, duplicate names, unsupported
   compression, oversized members, and suspicious expansion ratios;
6. extracts through a staging directory and atomically installs `data/ml-100k/`;
7. writes a local provenance marker so verified reruns are idempotent.

Both the archive and extracted directory are ignored by Git. A corrupt or partial
archive is never used; it is replaced only after a newly downloaded temporary file
passes every integrity check. An existing unverified extracted directory is never
overwritten automatically.

The exact extracted-file pins live next to the acquisition code in
`download_data.py`. `python run_demo.py --publish-reports` revalidates those files,
their local acquisition manifest, and the expected dimensions before it can
replace the checked aggregate reports. Ordinary and quick runs write only to the
ignored local artifact directory.

## Files validated and used

- The model consumes `u.data` (`user_id`, `movie_id`, rating, timestamp) and
  `u.item` (movie title, release information, and 19 genre indicators).
- Acquisition also verifies pinned content digests for `README`, `u.genre`, and
  `u.info`, but the model does not consume them.

The source archive also contains age, gender, occupation, and ZIP-code fields tied
to numbered user IDs, plus predefined splits. This project deliberately does not
load `u.user` or those demographics. Their presence is one reason not to describe
the complete source data as anonymous, and it does not remove the need for careful
privacy, fairness, and evaluation decisions.

## Fingerprints recorded by the experiment

The public report's parsed-input fingerprint is not the archive SHA-256 above. It
is a combined SHA-256 over the ordered `u.data` and `u.item` filenames, byte
lengths, and individual SHA-256 content digests consumed by the parser. The
archive checksum binds the downloaded ZIP; the parsed-input fingerprint binds the
two files used by the model. Neither fingerprint proves that a particular source-
code revision produced the report, so code provenance is recorded separately.

## Citation

GroupLens asks users to cite the following dataset paper:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. *ACM Transactions on Interactive Intelligent Systems*,
> 5(4), Article 19. <https://doi.org/10.1145/2827872>

This project is independent and is not endorsed by GroupLens or the University of
Minnesota.
