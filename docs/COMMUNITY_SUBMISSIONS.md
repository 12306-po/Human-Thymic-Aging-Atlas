# Community Submission v1

## Evidence layers

1. **Official Atlas** — the immutable manuscript-aligned `v1.0-18donor`
   release from GSE231906. The D-010 main results and `release/` files are not
   changed by community input.
2. **External evidence** — independently assessed cohort transfer or support.
   HRA007984 remains supplementary feasibility code, without scored external
   predictions or a validation claim.
3. **Community datasets** — contributor-provided result tables, separate from
   the Official Atlas and not published by this v1 feature.

## Current workflow

Open **Contribute data** in the Streamlit tab navigation. Upload the three
standardized CSV files described in [`schemas/README.md`](../schemas/README.md),
using the three template downloads on the page,
resolve any cell-type labels, confirm authorization and de-identification, then
select **Validate and prepare submission package**. The page assigns an ID and
shows **Submitted (this session) → Automated validation → Pending manual review
(package ready; not delivered)** when checks pass. Failed checks end at **Needs
correction**. Download the ZIP containing normalized CSVs, the mapping table,
and `validation_report.json`.

The package is a local export, not a server receipt. No reviewer is notified,
no database or object store is configured, and no dataset is automatically
published. Data and report remain in Streamlit session memory until the session
ends or the user clears them. The browser's upload mechanism and deployment
infrastructure may handle bytes transiently, so deploy only in an environment
appropriate for data that can be shared openly.

Automated checks cover schema, blanks, donor identity and coverage, age range,
cell labels, finite beta, q range, and integer `n_donors`. They cannot verify
consent, de-identification, sample overlap, biological equivalence, or model
quality. Manual review and an explicit versioned curation step are required
before any future Community Atlas listing. Community data must never mutate
the Official Atlas release or be silently merged into manuscript analyses.

## Future persistence boundary

`submission_store.py` contains the session-only storage adapter. A future
backend may implement authenticated intake, private object storage, virus
scanning, quota and abuse limits, retention/deletion, access controls, audit
logs, reviewer decisions, and explicit publish actions. It must remain
separate from `release/`. No Supabase, S3, or other credentials are needed for
v1; no environment variables are required for this feature.
