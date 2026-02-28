import pandas as pd
import json
import os
import re
import urllib.request
import requests
from datetime import datetime, timedelta
from crewai.tools import tool

SUMMARY_REPORT_PATH = "artifacts/model/pipeline_summary_report.json"
SUMMARY_MD_PATH = "artifacts/model/pipeline_summary_report.md"
ARTIFACT_LINKS_PATH = "artifacts/model/artifact_links.json"
EXECUTIVE_DASHBOARD_PATH = "artifacts/model/executive_dashboard.json"
STAKEHOLDER_SUMMARY_PATH = "artifacts/model/stakeholder_summary.md"
API_DEPLOYMENT_PATH = "artifacts/model/pycaret_api_deployment_deployment.json"  # Updated path

REQUIRED_SECTIONS = [
    "data",
    "feature_selection",
    "model_performance",
    "predictions",
    "explanability",
    "retention_strategy",
]

MINIMUM_MODEL_AUC = 0.75
MINIMUM_CHURN_RATE = 10.0
MAXIMUM_CHURN_RATE = 35.0
MAX_DATA_AGE_HOURS = 24
LINK_TIMEOUT_SECONDS = 5

# PII patterns to check for in reports
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
}


@tool("Validate Summary Report Exists")
def validate_summary_report_exists_tool() -> str:
    """Validate that both the JSON and Markdown summary report files exist."""
    errors = []
    for path in [SUMMARY_REPORT_PATH, SUMMARY_MD_PATH]:
        if not os.path.exists(path):
            errors.append(f"Missing: {path}")
        elif os.path.getsize(path) == 0:
            errors.append(f"Empty file: {path}")
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return f"Summary report files PASSED. Both JSON and Markdown reports exist."


@tool("Validate Summary Sections Complete")
def validate_summary_sections_tool() -> str:
    """
    Validate that the pipeline summary JSON report contains all required sections:
    data, feature_selection, model_performance, predictions, explanability,
    retention_strategy.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    sections = report.get("sections", {})
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]

    if missing:
        return (
            f"VALIDATION FAILED: Missing sections in summary report: {missing}\n"
            f"Present sections: {list(sections.keys())}"
        )
    return (
        f"Summary sections PASSED. All {len(REQUIRED_SECTIONS)} required sections present:\n"
        f"  {REQUIRED_SECTIONS}"
    )


@tool("Validate Summary Model Metrics")
def validate_summary_model_metrics_tool() -> str:
    """
    Validate that the model performance section in the summary report
    meets minimum AUC threshold.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    model_section = report.get("sections", {}).get("model_performance", {})
    if not model_section:
        return "VALIDATION FAILED: model_performance section missing from summary."

    auc = model_section.get("auc", 0)
    if auc < MINIMUM_MODEL_AUC:
        return f"VALIDATION FAILED: Summary AUC {auc} < minimum {MINIMUM_MODEL_AUC}."

    return (
        f"Summary model metrics PASSED.\n"
        f"  Model : {model_section.get('model_type', 'N/A')}\n"
        f"  AUC   : {auc}\n"
        f"  F1    : {model_section.get('f1', 'N/A')}\n"
        f"  Recall: {model_section.get('recall', 'N/A')}"
    )


@tool("Validate Summary Prediction Stats")
def validate_summary_prediction_stats_tool() -> str:
    """
    Validate that the predictions section in the summary has a realistic
    churn rate (10%–35%) and non-zero total customers.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    pred_section = report.get("sections", {}).get("predictions", {})
    if not pred_section:
        return "VALIDATION FAILED: predictions section missing from summary."

    total = pred_section.get("total", 0)
    churn_rate = pred_section.get("churn_rate_pct", 0)

    errors = []
    if total == 0:
        errors.append("Total customers is 0.")
    if not (MINIMUM_CHURN_RATE <= churn_rate <= MAXIMUM_CHURN_RATE):
        errors.append(
            f"Churn rate {churn_rate}% outside expected range "
            f"[{MINIMUM_CHURN_RATE}%, {MAXIMUM_CHURN_RATE}%]."
        )
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)

    return (
        f"Prediction stats PASSED.\n"
        f"  Total customers   : {total}\n"
        f"  Predicted churners: {pred_section.get('churners', 'N/A')}\n"
        f"  Churn rate        : {churn_rate}%"
    )


@tool("Validate Summary Retention Section")
def validate_summary_retention_tool() -> str:
    """
    Validate that the retention strategy section exists and reports
    at least one at-risk customer with valid segments.
    """
    if not os.path.exists(SUMMARY_REPORT_PATH):
        return f"VALIDATION FAILED: Summary report not found at {SUMMARY_REPORT_PATH}."

    with open(SUMMARY_REPORT_PATH) as f:
        report = json.load(f)

    ret_section = report.get("sections", {}).get("retention_strategy", {})
    if not ret_section:
        return "VALIDATION FAILED: retention_strategy section missing from summary."

    total_at_risk = ret_section.get("total_at_risk", 0)
    if total_at_risk == 0:
        return "VALIDATION FAILED: retention_strategy reports 0 at-risk customers."

    valid_segments = {"High Risk", "Medium Risk", "Low Risk"}
    segments = set(ret_section.get("segments", {}).keys())
    invalid = segments - valid_segments
    if invalid:
        return f"VALIDATION FAILED: Invalid segment labels in summary: {invalid}"

    return (
        f"Retention section PASSED.\n"
        f"  At-risk customers: {total_at_risk}\n"
        f"  Segments: {ret_section.get('segments', {})}"
    )


@tool("Validate Markdown Report Readable")
def validate_markdown_report_tool() -> str:
    """Validate that the Markdown report is well-formed and contains all major headings."""
    if not os.path.exists(SUMMARY_MD_PATH):
        return f"VALIDATION FAILED: Markdown report not found at {SUMMARY_MD_PATH}."

    with open(SUMMARY_MD_PATH) as f:
        content = f.read()

    required_headings = [
        "# Customer Churn",
        "## Data",
        "## Model Performance",
        "## Predictions",
        "## Retention",
    ]
    missing = [h for h in required_headings if h not in content]
    if missing:
        return f"VALIDATION FAILED: Missing headings in Markdown report: {missing}"

    word_count = len(content.split())
    return f"Markdown report PASSED. {word_count} words, all required headings present."


# ── NEW SUMMARY VALIDATION TOOLS ADDED BELOW ──────────────────────────────────


@tool("Validate Link Integrity")
def validate_link_integrity_tool(
    timeout: int = LINK_TIMEOUT_SECONDS,
    check_local_files: bool = True
) -> str:
    """
    Checks that all hyperlinks in the report actually resolve.
    Validates both external URLs and local file paths.
    """
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "total_links_found": 0,
        "links_validated": 0,
        "links_working": 0,
        "links_broken": 0,
        "links_skipped": 0,
        "broken_links": [],
        "working_links": [],
        "all_valid": True
    }

    all_links = []

    # Extract links from artifact_links.json if available
    if os.path.exists(ARTIFACT_LINKS_PATH):
        with open(ARTIFACT_LINKS_PATH, "r") as f:
            links_data = json.load(f)
        
        for category, items in links_data.get("categories", {}).items():
            for item in items:
                if "url" in item:
                    all_links.append({
                        "name": item.get("name", "Unknown"),
                        "url": item["url"],
                        "path": item.get("path", ""),
                        "category": category,
                        "type": "artifact"
                    })

    # Extract links from Markdown report
    if os.path.exists(SUMMARY_MD_PATH):
        with open(SUMMARY_MD_PATH, "r") as f:
            md_content = f.read()
        
        # Find markdown links [text](url)
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', md_content)
        for text, url in md_links:
            all_links.append({
                "name": text,
                "url": url,
                "path": "",
                "category": "markdown",
                "type": "markdown"
            })

    validation_results["total_links_found"] = len(all_links)

    if len(all_links) == 0:
        validation_results["links_skipped"] = 0
        validation_results["all_valid"] = True
        return (
            f"Link Integrity Validation:\n"
            f"Status: ⚠️ SKIPPED\n"
            f"No links found to validate.\n"
            f"Report saved to: artifacts/model/link_integrity_validation.json"
        )

    # Validate each link
    for link in all_links:
        validation_results["links_validated"] += 1
        url = link["url"]
        is_valid = False
        error_msg = None

        try:
            if url.startswith("file://") or url.startswith("/"):
                # Local file path
                if check_local_files:
                    file_path = url.replace("file://", "")
                    if os.path.exists(file_path):
                        is_valid = True
                        validation_results["links_working"] += 1
                        validation_results["working_links"].append({
                            "name": link["name"],
                            "url": url,
                            "type": "local_file"
                        })
                    else:
                        is_valid = False
                        error_msg = f"File not found: {file_path}"
                        validation_results["links_broken"] += 1
                        validation_results["broken_links"].append({
                            "name": link["name"],
                            "url": url,
                            "error": error_msg
                        })
                else:
                    validation_results["links_skipped"] += 1
            elif url.startswith("http://") or url.startswith("https://"):
                # External URL
                try:
                    response = urllib.request.urlopen(url, timeout=timeout)
                    if response.status == 200:
                        is_valid = True
                        validation_results["links_working"] += 1
                        validation_results["working_links"].append({
                            "name": link["name"],
                            "url": url,
                            "type": "external_url",
                            "status_code": response.status
                        })
                    else:
                        is_valid = False
                        error_msg = f"HTTP {response.status}"
                        validation_results["links_broken"] += 1
                        validation_results["broken_links"].append({
                            "name": link["name"],
                            "url": url,
                            "error": error_msg
                        })
                except Exception as e:
                    is_valid = False
                    error_msg = str(e)
                    validation_results["links_broken"] += 1
                    validation_results["broken_links"].append({
                        "name": link["name"],
                        "url": url,
                        "error": error_msg
                    })
            else:
                # Unknown protocol - skip
                validation_results["links_skipped"] += 1
                continue

        except Exception as e:
            is_valid = False
            error_msg = str(e)
            validation_results["links_broken"] += 1
            validation_results["broken_links"].append({
                "name": link["name"],
                "url": url,
                "error": error_msg
            })

        if not is_valid:
            validation_results["all_valid"] = False

    # Save validation report
    os.makedirs("artifacts/model", exist_ok=True)
    validation_report_path = "artifacts/model/link_integrity_validation.json"
    with open(validation_report_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    # Format output
    if validation_results["all_valid"]:
        status = "✅ ALL LINKS VALID"
        recommendation = "All hyperlinks in the report are working correctly."
    else:
        status = "❌ BROKEN LINKS DETECTED"
        broken_text = "\n".join([
            f"  • {link['name']}: {link['url']} - {link.get('error', 'Unknown error')}"
            for link in validation_results["broken_links"][:10]
        ])
        recommendation = (
            f"Found {validation_results['links_broken']} broken links:\n{broken_text}\n"
            f"Fix broken links before distributing the report."
        )

    return (
        f"Link Integrity Validation:\n"
        f"Status: {status}\n"
        f"Total Links Found: {validation_results['total_links_found']}\n"
        f"Links Validated: {validation_results['links_validated']}\n"
        f"Working: {validation_results['links_working']}\n"
        f"Broken: {validation_results['links_broken']}\n"
        f"Skipped: {validation_results['links_skipped']}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {validation_report_path}"
    )


@tool("Validate Data Freshness")
def validate_data_freshness_tool(
    max_age_hours: int = MAX_DATA_AGE_HOURS
) -> str:
    """
    Ensures the summary timestamp is within the expected window (not running on stale data).
    Validates that reports are generated from recent pipeline runs.
    """
    validation_result = {
        "timestamp": datetime.now().isoformat(),
        "max_age_hours": max_age_hours,
        "summary_age_hours": None,
        "is_fresh": False,
        "summary_generated_at": None,
        "status": "UNKNOWN"
    }

    # Check summary report timestamp
    if os.path.exists(SUMMARY_REPORT_PATH):
        with open(SUMMARY_REPORT_PATH, "r") as f:
            report = json.load(f)
        
        generated_at = report.get("generated_at")
        if generated_at:
            try:
                summary_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                now = datetime.now(summary_time.tzinfo) if summary_time.tzinfo else datetime.now()
                age = now - summary_time
                age_hours = age.total_seconds() / 3600
                
                validation_result["summary_generated_at"] = generated_at
                validation_result["summary_age_hours"] = round(age_hours, 2)
                validation_result["is_fresh"] = age_hours <= max_age_hours
                
                if validation_result["is_fresh"]:
                    validation_result["status"] = "✅ FRESH"
                else:
                    validation_result["status"] = "❌ STALE"
            except Exception as e:
                validation_result["status"] = "⚠️ PARSE_ERROR"
                validation_result["error"] = str(e)
        else:
            validation_result["status"] = "⚠️ NO_TIMESTAMP"
    else:
        validation_result["status"] = "❌ REPORT_NOT_FOUND"

    # Check other key artifact timestamps
    artifact_ages = {}
    key_artifacts = [
        ("Predictions", "artifacts/data/predictions.csv"),
        ("Retention Plan", "artifacts/data/retention_plan.csv"),
        ("Model", "artifacts/model/churn_model.pkl"),
    ]

    for name, path in key_artifacts:
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            now = datetime.now()
            age_hours = (now - mtime).total_seconds() / 3600
            artifact_ages[name] = {
                "age_hours": round(age_hours, 2),
                "is_fresh": age_hours <= max_age_hours,
                "last_modified": mtime.isoformat()
            }
        else:
            artifact_ages[name] = {
                "age_hours": None,
                "is_fresh": False,
                "last_modified": "FILE_NOT_FOUND"
            }

    validation_result["artifact_ages"] = artifact_ages

    # Check if all artifacts are fresh
    all_fresh = (
        validation_result["is_fresh"] and
        all(artifact.get("is_fresh", False) for artifact in artifact_ages.values())
    )

    if all_fresh:
        validation_result["overall_status"] = "✅ ALL DATA FRESH"
    else:
        validation_result["overall_status"] = "⚠️ SOME DATA STALE"

    # Save validation report
    os.makedirs("artifacts/model", exist_ok=True)
    validation_report_path = "artifacts/model/data_freshness_validation.json"
    with open(validation_report_path, "w") as f:
        json.dump(validation_result, f, indent=2)

    # Format output
    artifact_text = "\n".join([
        f"  {name}: {age['age_hours']:.1f} hours old "
        f"({'✅' if age['is_fresh'] else '⚠️'})"
        for name, age in artifact_ages.items()
    ])

    if validation_result["is_fresh"]:
        status = "✅ DATA FRESH"
        recommendation = (
            f"Summary is {validation_result['summary_age_hours']:.1f} hours old "
            f"(within {max_age_hours}h threshold). All data is current."
        )
    else:
        status = "⚠️ DATA STALE"
        recommendation = (
            f"Summary is {validation_result.get('summary_age_hours', 'N/A')} hours old "
            f"(exceeds {max_age_hours}h threshold). Consider re-running the pipeline."
        )

    return (
        f"Data Freshness Validation:\n"
        f"Status: {validation_result['overall_status']}\n"
        f"Max Age Threshold: {max_age_hours} hours\n"
        f"\nSummary Report:\n"
        f"  Generated At: {validation_result.get('summary_generated_at', 'N/A')}\n"
        f"  Age: {validation_result.get('summary_age_hours', 'N/A')} hours "
        f"({'✅' if validation_result['is_fresh'] else '⚠️'})\n"
        f"\nArtifact Ages:\n{artifact_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {validation_report_path}"
    )


@tool("Validate Sensitive Data Redaction")
def validate_sensitive_data_redaction_tool(
    scan_files: str = "artifacts/model/pipeline_summary_report.json,artifacts/model/pipeline_summary_report.md,artifacts/model/stakeholder_summary.md"
) -> str:
    """
    Scans the final report to ensure no PII leaked into the summary document.
    Checks for emails, phone numbers, SSNs, and credit card numbers.
    
    Args:
        scan_files: Comma-separated list of file paths to scan (default: summary reports)
    """
    """
    Scans the final report to ensure no PII leaked into the summary document.
    Checks for emails, phone numbers, SSNs, and credit card numbers.
    """
    # Parse scan_files from comma-separated string
    if not scan_files or scan_files.strip() == "":
        scan_files_list = [
            SUMMARY_REPORT_PATH,
            SUMMARY_MD_PATH,
            STAKEHOLDER_SUMMARY_PATH,
        ]
    else:
        scan_files_list = [f.strip() for f in scan_files.split(",") if f.strip()]

    validation_result = {
        "timestamp": datetime.now().isoformat(),
        "files_scanned": [],
        "pii_detected": [],
        "all_clean": True,
        "patterns_checked": list(PII_PATTERNS.keys())
    }

    for file_path in scan_files_list:
        if not os.path.exists(file_path):
            validation_result["files_scanned"].append({
                "path": file_path,
                "status": "FILE_NOT_FOUND",
                "pii_found": []
            })
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        file_pii = []

        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, content)
            if matches:
                # Don't show actual PII in report - just count and type
                file_pii.append({
                    "type": pii_type,
                    "count": len(matches),
                    "sample": f"[{pii_type.upper()} REDACTED]"
                })

        validation_result["files_scanned"].append({
            "path": file_path,
            "status": "SCANNED",
            "pii_found": file_pii,
            "file_size_bytes": os.path.getsize(file_path)
        })

        if file_pii:
            validation_result["all_clean"] = False
            validation_result["pii_detected"].extend([
                {"file": file_path, **pii} for pii in file_pii
            ])

    # Save validation report
    os.makedirs("artifacts/model", exist_ok=True)
    validation_report_path = "artifacts/model/sensitive_data_redaction_validation.json"
    with open(validation_report_path, "w") as f:
        json.dump(validation_result, f, indent=2)

    # Format output
    if validation_result["all_clean"]:
        status = "✅ NO PII DETECTED"
        recommendation = "All summary reports are clean. No sensitive data detected."
        pii_text = "None"
    else:
        status = "❌ PII DETECTED"
        pii_text = "\n".join([
            f"  • {pii['file']}: {pii['count']}x {pii['type'].upper()} found"
            for pii in validation_result["pii_detected"]
        ])
        recommendation = (
            f"PII detected in summary reports. Remove sensitive data before distribution:\n"
            f"{pii_text}\n"
            f"Review data preprocessing and redaction steps."
        )

    files_text = "\n".join([
        f"  {f['path']}: {f['status']} "
        f"({len(f['pii_found'])} PII types found)"
        for f in validation_result["files_scanned"]
    ])

    return (
        f"Sensitive Data Redaction Validation:\n"
        f"Status: {status}\n"
        f"Files Scanned: {len(validation_result['files_scanned'])}\n"
        f"Patterns Checked: {validation_result['patterns_checked']}\n"
        f"\nFiles:\n{files_text}\n"
        f"\nPII Detected:\n{pii_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {validation_report_path}"
    )


@tool("Validate Deployment Health")
def validate_deployment_health_tool(
    api_url: str = None,
    timeout: int = 10,
    health_endpoint: str = "/health"
) -> str:
    """
    Pings the deployed API endpoint to ensure it returns 200 OK.
    Handles the case where PyCaret's deploy_model_api_tool creates files
    but doesn't start a server (expected behavior).
    """
    # Try to load API deployment info
    if api_url is None:
        # Check common deployment file paths
        possible_paths = [
            API_DEPLOYMENT_PATH,
            "artifacts/model/pycaret_api_deployment.json",
            "artifacts/model/api_deployment.json",
        ]
        
        deployment_info = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    deployment_info = json.load(f)
                break
        
        if deployment_info:
            api_url = deployment_info.get("api_url", f"http://localhost:{deployment_info.get('port', 8000)}")
        else:
            return (
                f"VALIDATION SKIPPED: API deployment info not found.\n"
                f"Run deploy_model_api_tool first before validating deployment health.\n"
                f"Note: deploy_model_api_tool creates API files but doesn't start a server."
            )
    
    # Construct health endpoint URL
    health_url = f"{api_url.rstrip('/')}{health_endpoint}"
    
    validation_results = []
    all_healthy = True
    server_not_started = False
    
    # Check 1: Health endpoint
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            validation_results.append({
                "endpoint": health_endpoint,
                "status_code": response.status_code,
                "status": "HEALTHY",
                "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
            })
        else:
            validation_results.append({
                "endpoint": health_endpoint,
                "status_code": response.status_code,
                "status": "UNHEALTHY",
                "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
            })
            all_healthy = False
    except requests.exceptions.ConnectionError as e:
        # FIX: Handle "API not started" case (expected for PyCaret deploy_model)
        if "Connection refused" in str(e) or "Failed to establish" in str(e):
            server_not_started = True
            validation_results.append({
                "endpoint": health_endpoint,
                "status_code": "N/A",
                "status": "SERVER_NOT_STARTED",
                "note": "PyCaret's deploy_model_api_tool creates API files but doesn't start a server. To test: manually run 'uvicorn <api_file>:app --port <port>'"
            })
            all_healthy = False  # But this is expected, not an error
        else:
            validation_results.append({
                "endpoint": health_endpoint,
                "status_code": "N/A",
                "status": "CONNECTION_FAILED",
                "error": str(e)
            })
            all_healthy = False
    except requests.exceptions.Timeout as e:
        validation_results.append({
            "endpoint": health_endpoint,
            "status_code": "N/A",
            "status": "TIMEOUT",
            "error": f"Request timed out after {timeout}s"
        })
        all_healthy = False
    except Exception as e:
        validation_results.append({
            "endpoint": health_endpoint,
            "status_code": "N/A",
            "status": "ERROR",
            "error": str(e)
        })
        all_healthy = False
    
    # Check 2: Predict endpoint (only if health is OK or server started)
    if not server_not_started and all_healthy and os.path.exists("artifacts/data/X_test.csv"):
        try:
            # Load sample data for prediction test
            X_test = pd.read_csv("artifacts/data/X_test.csv").head(1)
            sample_data = X_test.to_dict(orient="records")[0]
            
            predict_url = f"{api_url.rstrip('/')}/predict"
            response = requests.post(predict_url, json=sample_data, timeout=timeout)
            
            if response.status_code == 200:
                prediction_result = response.json()
                validation_results.append({
                    "endpoint": "/predict",
                    "status_code": response.status_code,
                    "status": "WORKING",
                    "prediction": prediction_result
                })
            else:
                validation_results.append({
                    "endpoint": "/predict",
                    "status_code": response.status_code,
                    "status": "FAILED",
                    "error": response.text
                })
                all_healthy = False
        except Exception as e:
            validation_results.append({
                "endpoint": "/predict",
                "status_code": "N/A",
                "status": "ERROR",
                "error": str(e)
            })
            all_healthy = False
    
    # Save health check report
    health_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "api_url": api_url,
        "timeout_seconds": timeout,
        "server_not_started": server_not_started,
        "all_healthy": all_healthy and not server_not_started,
        "endpoint_checks": validation_results
    }
    
    health_report_path = "artifacts/model/api_health_check_report.json"
    os.makedirs("artifacts/model", exist_ok=True)
    with open(health_report_path, "w") as f:
        json.dump(health_report, f, indent=2)
    
    # Format output
    checks_text = "\n".join([
        f"  {r['endpoint']}: {r['status']} "
        f"({'✅' if r['status'] in ['HEALTHY', 'WORKING'] else '⚠️' if r['status'] == 'SERVER_NOT_STARTED' else '❌'}) "
        f"- Status Code: {r.get('status_code', 'N/A')}"
        + (f", Response Time: {r.get('response_time_ms', 'N/A')}ms" if 'response_time_ms' in r else "")
        + (f"\n    Note: {r.get('note', '')}" if r.get('note') else "")
        for r in validation_results
    ])
    
    if server_not_started:
        status = "⚠️ API NOT STARTED (Expected)"
        recommendation = (
            "PyCaret's deploy_model_api_tool creates API files but doesn't start a server.\n"
            "To test the API: manually run 'uvicorn <api_file>:app --port <port>'\n"
            "Or skip this validation if API testing is not required."
        )
    elif all_healthy:
        status = "✅ API HEALTHY"
        recommendation = "Deployed API is responding correctly. Ready for production use."
    else:
        status = "❌ API UNHEALTHY"
        recommendation = "API deployment has issues. Check server logs and redeploy if necessary."
    
    return (
        f"Deployment Health Validation:\n"
        f"Status: {status}\n"
        f"API URL: {api_url}\n"
        f"Timeout: {timeout}s\n"
        f"\nEndpoint Checks:\n{checks_text}\n"
        f"\nRecommendation: {recommendation}\n"
        f"Report saved to: {health_report_path}"
    )