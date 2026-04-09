"""
Ariston Agent Swarm — Master Multi-Agent Orchestrator.

Chains all available agents into a unified intelligence loop:

  1. PatientHistoryAgent       — inject longitudinal patient context
  2. IntentClassifier           — classify query and select routing path
  3. PharmacogenomicsAgent      — PGx gene/drug interaction alerts
  4. DigitalTwinAgent           — in-silico treatment simulation
  5. IoMTAgent                  — device adherence forecasting
  6. ClinicalPipeline           — FHIR-grounded structured decision
  7. RegulatoryCopilot          — GxP-compliant ACR generation
  8. BenchmarkLogger            — score the swarm output

All sub-agent results are aggregated into a unified SwarmReport.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.classifier import classifier
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.agent.twin_agent import digital_twin_agent
from vinci_core.agent.iomt_agent import iomt_agent
from vinci_core.agent.regulatory_agent import regulatory_copilot
from vinci_core.workflows.clinical_pipeline import clinical_pipeline
from vinci_core.engine import engine
from vinci_core.evaluation.benchmark_logger import benchmark_logger


class AgentSwarm:
    """
    Coordinates all Ariston agents in a multi-stage reasoning loop.

    Each stage is independent and its result feeds into the next,
    creating an enriched context that grows richer with each agent pass.
    """

    async def run(
        self,
        prompt: str,
        patient_id: Optional[str] = None,
        drug_name: Optional[str] = None,
        fhir_bundle: Optional[List[Dict[str, Any]]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        genetics: Optional[List[str]] = None,
        include_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full agent swarm loop.

        Args:
            prompt: The clinical / pharma / data query
            patient_id: Optional patient ID for longitudinal history injection
            drug_name: Optional drug name for PGx + twin simulation
            fhir_bundle: Optional FHIR resources for clinical pipeline
            telemetry: Optional IoMT device telemetry dict
            genetics: Optional list of known genetic variants (e.g. ["CYP2D6 Poor Metabolizer"])
            include_stages: Optional list of stages to run (default: all).
                            Choices: patient, pgx, twin, iomt, clinical, regulatory

        Returns:
            SwarmReport dict with per-agent results, aggregated insights, and GxP report.
        """
        swarm_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        stages_run: List[str] = []
        stages = include_stages or ["patient", "classifier", "pgx", "twin", "iomt", "clinical", "regulatory"]

        report: Dict[str, Any] = {
            "swarm_id": swarm_id,
            "prompt": prompt,
            "patient_id": patient_id,
            "drug_name": drug_name,
            "started_at": started_at,
            "stages": {},
        }

        # ── Stage 1: Patient history injection ────────────────────────────────
        patient_history = ""
        if "patient" in stages and patient_id:
            patient_history = patient_agent.get_full_history(patient_id)
            report["stages"]["patient"] = {
                "status": "ok",
                "history_length": len(patient_history),
                "has_history": bool(patient_history),
            }
            stages_run.append("patient")

        # ── Stage 2: Intent classification ────────────────────────────────────
        detected_layer = "clinical"
        if "classifier" in stages:
            try:
                detected_layer = await classifier.classify(prompt)
                report["stages"]["classifier"] = {
                    "status": "ok",
                    "detected_layer": detected_layer,
                }
                stages_run.append("classifier")
            except Exception as e:
                report["stages"]["classifier"] = {"status": "error", "error": str(e)}

        # ── Stage 3: PGx cross-reference ──────────────────────────────────────
        pgx_result: Dict[str, Any] = {}
        if "pgx" in stages and drug_name:
            try:
                pgx_result = await pharmacogenomics_agent.cross_reference(drug_name, patient_id)
                report["stages"]["pgx"] = {
                    "status": "ok",
                    "drug": drug_name,
                    "genomic_status": pgx_result.get("status", "unknown"),
                    "alert_count": len(pgx_result.get("genomic_alerts", [])),
                    "alerts": pgx_result.get("genomic_alerts", []),
                    "recommendation": pgx_result.get("recommendation", ""),
                }
                stages_run.append("pgx")
            except Exception as e:
                report["stages"]["pgx"] = {"status": "error", "error": str(e)}

        # ── Stage 4: Digital Twin simulation ──────────────────────────────────
        twin_result: Dict[str, Any] = {}
        if "twin" in stages and drug_name:
            try:
                history_for_twin = patient_history or prompt
                gene_list = genetics or []
                twin_result = digital_twin_agent.simulate_treatment(
                    history=history_for_twin,
                    drug=drug_name,
                    genetics=gene_list,
                )
                report["stages"]["twin"] = {
                    "status": "ok",
                    **twin_result,
                }
                stages_run.append("twin")
            except Exception as e:
                report["stages"]["twin"] = {"status": "error", "error": str(e)}

        # ── Stage 5: IoMT adherence forecast ──────────────────────────────────
        if "iomt" in stages:
            try:
                iomt_result = iomt_agent.forecast_adherence(
                    history=patient_history or prompt,
                    telemetry=telemetry,
                )
                report["stages"]["iomt"] = {
                    "status": "ok",
                    **iomt_result,
                }
                stages_run.append("iomt")
            except Exception as e:
                report["stages"]["iomt"] = {"status": "error", "error": str(e)}

        # ── Stage 6: Clinical Pipeline (FHIR + consensus) ─────────────────────
        clinical_response = None
        if "clinical" in stages:
            try:
                # Build enriched prompt from all prior stages
                enriched_prompt = _build_enriched_prompt(
                    prompt=prompt,
                    patient_history=patient_history,
                    pgx_result=pgx_result,
                    twin_result=twin_result,
                )
                clinical_response = await clinical_pipeline.execute(
                    prompt=enriched_prompt,
                    fhir_bundle=fhir_bundle,
                    patient_id=patient_id,
                )
                report["stages"]["clinical"] = {
                    "status": "ok",
                    "content": clinical_response.content,
                    "model": clinical_response.model,
                    "safety": clinical_response.metadata.get("safety", {}),
                    "grounded_entities": clinical_response.metadata.get("grounded_entities", []),
                }
                stages_run.append("clinical")
            except Exception as e:
                # Fallback to direct engine run if pipeline fails
                try:
                    fallback = await engine.run(
                        prompt=prompt,
                        layer=detected_layer,
                        patient_id=patient_id,
                    )
                    report["stages"]["clinical"] = {
                        "status": "fallback",
                        "content": fallback.content,
                        "model": fallback.model,
                        "safety": fallback.metadata.get("safety", {}),
                        "error": str(e),
                    }
                    clinical_response = fallback
                    stages_run.append("clinical")
                except Exception as e2:
                    report["stages"]["clinical"] = {"status": "error", "error": str(e2)}

        # ── Stage 7: Regulatory Copilot — GxP report ─────────────────────────
        if "regulatory" in stages:
            try:
                audit_logs = []  # could inject actual audit trail
                content_for_report = (
                    clinical_response.content if clinical_response else prompt
                )
                gxp_report = regulatory_copilot.generate_report(
                    job_id=swarm_id,
                    prompt=prompt,
                    result=content_for_report,
                    audit_logs=audit_logs,
                )
                report["stages"]["regulatory"] = {
                    "status": "ok",
                    "gxp_report": gxp_report,
                }
                stages_run.append("regulatory")
            except Exception as e:
                report["stages"]["regulatory"] = {"status": "error", "error": str(e)}

        # ── Swarm summary ──────────────────────────────────────────────────────
        completed_at = datetime.now(timezone.utc).isoformat()
        report["stages_run"] = stages_run
        report["stages_requested"] = stages
        report["completed_at"] = completed_at
        report["summary"] = _build_summary(report)

        # ── Benchmark logging ──────────────────────────────────────────────────
        clinical_stage = report["stages"].get("clinical", {})
        if clinical_stage.get("content"):
            benchmark_logger.evaluate_and_log(
                prompt=prompt,
                response_content=clinical_stage["content"],
                response_metadata={
                    "safety": clinical_stage.get("safety", {}),
                    "model": clinical_stage.get("model", "swarm"),
                    "rag_used": False,
                    "consensus": True,
                },
                layer="swarm",
            )

        return report


def _build_enriched_prompt(
    prompt: str,
    patient_history: str,
    pgx_result: Dict[str, Any],
    twin_result: Dict[str, Any],
) -> str:
    """Build a context-enriched prompt incorporating all prior agent outputs."""
    parts = [prompt]

    if patient_history:
        parts.append(f"\n{patient_history}")

    if pgx_result.get("genomic_alerts"):
        alerts = pgx_result["genomic_alerts"]
        alert_lines = [
            f"  [{a['severity']}] {a['gene']}: {a['risk_description']}"
            for a in alerts
        ]
        parts.append(f"\nPHARMACOGENOMIC ALERTS:\n" + "\n".join(alert_lines))

    if twin_result.get("prediction"):
        parts.append(
            f"\nDIGITAL TWIN SIMULATION:\n"
            f"  Prediction: {twin_result['prediction']}\n"
            f"  Efficacy: {twin_result.get('efficacy_score', 'N/A')}\n"
            f"  Toxicity Risk: {twin_result.get('toxicity_risk', 'N/A')}\n"
            f"  Organ Impact: {twin_result.get('organ_impact', {})}"
        )

    return "\n".join(parts)


def _build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a concise swarm summary for API consumers."""
    stages = report.get("stages", {})

    pgx = stages.get("pgx", {})
    twin = stages.get("twin", {})
    iomt = stages.get("iomt", {})
    clinical = stages.get("clinical", {})
    regulatory = stages.get("regulatory", {})

    return {
        "swarm_id": report["swarm_id"],
        "stages_completed": len(report.get("stages_run", [])),
        "clinical_response": clinical.get("content", ""),
        "clinical_model": clinical.get("model", ""),
        "safety_flag": clinical.get("safety", {}).get("flag", "SAFE"),
        "pgx_status": pgx.get("genomic_status", "not_run"),
        "pgx_alert_count": pgx.get("alert_count", 0),
        "twin_prediction": twin.get("prediction", "not_run"),
        "twin_efficacy": twin.get("efficacy_score"),
        "twin_toxicity": twin.get("toxicity_risk"),
        "iomt_adherence": iomt.get("adherence_score"),
        "iomt_risk": iomt.get("risk_level"),
        "gxp_report_generated": bool(regulatory.get("gxp_report")),
        "gxp_report": regulatory.get("gxp_report", ""),
    }


# Module-level singleton
agent_swarm = AgentSwarm()
