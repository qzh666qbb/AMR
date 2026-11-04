"""Human-quality workflow for the feasibility pilot."""
from __future__ import annotations

from pathlib import Path

from swan_feasibility2_common import load, sha256
from swan_feasibility2_generation import verify_documents
from swan_final1_quality import export_packets, finalize, verify_decision
from swan_final1_workflow import verify_inputs


def sources_for(folder):
    return load(Path(folder) / "sealed/pilot_sources.json")


def documents_for(folder):
    return verify_documents(folder, sources_for(folder))


def quality_records(documents):
    return [
        {
            "allocation_id": row["allocation_id"],
            "source_excerpt": row["source_excerpt"],
            "text": row["text"],
        }
        for row in documents
        if row["prose_structure_pass"]
    ]


def quality_action(folder, finalize_labels=False):
    folder = Path(folder)
    records = quality_records(documents_for(folder))
    destination = folder / "quality/pilot/baseline"
    contract = sha256(folder / "manifest.json")
    if finalize_labels:
        verify_inputs(destination, records)
        result = finalize(destination, contract, folder)
        return {
            "quality_pass_count": result["quality_pass_count"],
            "reviewed_items": len(records),
            "agreement": result["agreement"],
        }
    if (destination / "private_index.json").exists():
        verify_inputs(destination, records)
        return {"items_per_reviewer": len(records), "state": "already_exported"}
    return export_packets(records, destination, "baseline", contract)


def quality_decision(folder):
    folder = Path(folder)
    return verify_decision(
        folder / "quality/pilot/baseline",
        sha256(folder / "manifest.json"),
    )
