"""Import Stage03 gene effects only after verifying corrected-age provenance."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas_core_data import load_core


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def import_evidence(stage: Path, root: Path):
    target = root / "release/core_v1/gene_evidence"
    if target.exists():
        raise FileExistsError(f"Back up the current bundle before replacing it: {target}")
    provenance_path = stage / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (provenance.get("status") != "completed_corrected_age_effects_only"
            or provenance.get("donor17_corrected_age_years") != 1.75):
        raise ValueError("Stage03 was not completed with donor17 corrected to 1.75 years")
    donor_file = stage / "config/donor_metadata_master_corrected.csv"
    effects_file = stage / "data/GSE231906_study_specific_age_effects.csv"
    for label, path in (("donors", donor_file), ("gene_effects", effects_file)):
        if digest(path) != provenance.get("new_sha256", {}).get(label):
            raise ValueError(f"Stage03 {label} differs from its recorded provenance")
    site_donors, _ = load_core(root)
    donors = pd.read_csv(donor_file)
    donors = donors.loc[donors.study_id.astype(str).eq("GSE231906")].set_index("donor_id")
    expected = site_donors.set_index("donor_id").age_years
    if not donors.index.is_unique or set(donors.index) != set(expected.index):
        raise ValueError("Stage03 donor IDs do not match the website")
    if not np.allclose(donors.exact_age.reindex(expected.index), expected, atol=1e-8, rtol=0):
        raise ValueError("Stage03 ages differ from the website's corrected ages")
    frame = pd.read_csv(effects_file)
    required = {"study_id", "gene", "atlas_level2", "beta", "p", "q", "n_donors"}
    if not required.issubset(frame):
        raise ValueError("Gene effects lack the expected Stage03 columns")
    frame = frame.loc[frame.study_id.astype(str).eq("GSE231906")].copy()
    if frame.empty or frame.duplicated(["study_id", "atlas_level2", "gene"]).any():
        raise ValueError("Gene effects are empty or have duplicate gene/context rows")
    data = gzip.compress(frame.to_csv(index=False).encode("utf-8"), mtime=0)
    manifest = {"donor17_age_years": 1.75, "source_stage": str(stage),
                "source_provenance_sha256": digest(provenance_path),
                "source_effects_sha256": digest(effects_file),
                "effects_sha256": sha256(data).hexdigest(), "n_rows": len(frame),
                "scope": "Corrected donor-level gene age associations; HC3 stability is not implied"}
    target.mkdir(parents=True)
    (target / "gene_age_effects.csv.gz").write_bytes(data)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imported corrected gene evidence: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    import_evidence(args.stage, args.root)


if __name__ == "__main__":
    main()
