#!/usr/bin/env python3
"""
generate-immutable-rules-report.py
Fetches immutable-rules-gate.yml workflow run history via GitHub API
and generates a structured compliance report.
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dateutil import parser as dateutil_parser

WORKFLOW_FILENAME = "immutable-rules-gate.yml"
RULE_JOBS = {
    "rule01-zone-resilience":  "Rule-01: EventBus Zone Resilience",
    "rule02-pvc-zone-binding": "Rule-02: PVC Zone Binding Explicit",
    "rule03-policy-exception": "Rule-03: PolicyException Expiry",
    "rule04-flagger-sli":      "Rule-04: Flagger SLI Thresholds",
    "rule05-drill-cron":       "Rule-05: Chaos Drill Cron Active",
    "immutable-rules-summary": "Summary: All Rules",
}
MAX_SUMMARY_RUNS = 10


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token, repo):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.repo = repo

    def get(self, path, params=None):
        resp = self.session.get(f"{self.BASE}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_workflow_id(self):
        data = self.get(f"/repos/{self.repo}/actions/workflows")
        for wf in data.get("workflows", []):
            if wf.get("path", "").endswith(WORKFLOW_FILENAME):
                return wf["id"]
        return None

    def get_workflow_runs(self, workflow_id, since):
        runs = []
        page = 1
        while True:
            data = self.get(
                f"/repos/{self.repo}/actions/workflows/{workflow_id}/runs",
                params={"per_page": 100, "page": page,
                        "created": f">={since.strftime('%Y-%m-%d')}"},
            )
            batch = data.get("workflow_runs", [])
            if not batch:
                break
            for run in batch:
                created = dateutil_parser.parse(run["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if created >= since:
                    runs.append(run)
            if len(batch) < 100:
                break
            page += 1
        return runs

    def get_run_jobs(self, run_id):
        data = self.get(f"/repos/{self.repo}/actions/runs/{run_id}/jobs")
        return data.get("jobs", [])


def build_report(runs, client, actual_time, expected_time, drift_seconds, lookback_days, repo):
    rule_stats = {
        job_id: {"name": name, "pass": 0, "fail": 0, "skip": 0,
                 "last_status": "unknown", "last_run_at": None}
        for job_id, name in RULE_JOBS.items()
    }
    run_records = []
    total_pass = 0
    total_fail = 0

    for run in sorted(runs, key=lambda r: r["created_at"]):
        run_id = run["id"]
        run_conclusion = run.get("conclusion") or "in_progress"
        run_created = run["created_at"]
        run_url = run["html_url"]

        jobs = client.get_run_jobs(run_id)
        job_map = {}
        for job in jobs:
            job_name = job.get("name", "").lower().replace(" ", "-").replace(":", "")
            conclusion = job.get("conclusion") or "in_progress"
            for job_id in RULE_JOBS:
                if job_id in job_name:
                    job_map[job_id] = conclusion
                    break

        run_rule_results = {}
        for job_id in RULE_JOBS:
            conclusion = job_map.get(job_id, "skipped")
            run_rule_results[job_id] = conclusion
            if conclusion == "success":
                rule_stats[job_id]["pass"] += 1
                rule_stats[job_id]["last_status"] = "PASS"
            elif conclusion in ("failure", "timed_out", "cancelled"):
                rule_stats[job_id]["fail"] += 1
                rule_stats[job_id]["last_status"] = "FAIL"
            else:
                rule_stats[job_id]["skip"] += 1
            rule_stats[job_id]["last_run_at"] = run_created

        overall_run = ("PASS" if run_conclusion == "success"
                       else "FAIL" if run_conclusion in ("failure", "timed_out")
                       else "SKIP")
        if overall_run == "PASS":
            total_pass += 1
        elif overall_run == "FAIL":
            total_fail += 1

        run_records.append({
            "run_id": run_id,
            "run_url": run_url,
            "created_at": run_created,
            "conclusion": run_conclusion,
            "overall": overall_run,
            "rules": run_rule_results,
        })

    total_runs = len(run_records)
    pass_rate = round(total_pass / total_runs * 100, 1) if total_runs > 0 else 0.0
    overall_status = "PASS" if total_fail == 0 else "FAIL"

    rule_breakdown = []
    for job_id, stats in rule_stats.items():
        rule_breakdown.append({
            "job_id": job_id,
            "name": stats["name"],
            "pass": stats["pass"],
            "fail": stats["fail"],
            "skip": stats["skip"],
            "pass_rate_pct": round(stats["pass"] / total_runs * 100, 1) if total_runs > 0 else 0.0,
            "last_status": stats["last_status"],
            "last_run_at": stats["last_run_at"],
        })

    recent_failures = [
        {
            "run_id": r["run_id"],
            "run_url": r["run_url"],
            "created_at": r["created_at"],
            "failed_rules": [k for k, v in r["rules"].items()
                             if v in ("failure", "timed_out", "cancelled")],
        }
        for r in run_records if r["overall"] == "FAIL"
    ][-5:]

    return {
        "report_metadata": {
            "generated_at": actual_time,
            "expected_schedule_time": expected_time,
            "cron_drift_seconds": drift_seconds,
            "cron_drift_ok": abs(drift_seconds) <= 3600,
            "lookback_days": lookback_days,
            "repo": repo,
            "workflow": WORKFLOW_FILENAME,
        },
        "summary": {
            "overall_status": overall_status,
            "total_runs_analyzed": total_runs,
            "total_pass": total_pass,
            "total_failures": total_fail,
            "pass_rate_pct": pass_rate,
        },
        "rule_breakdown": rule_breakdown,
        "recent_failures": recent_failures,
        "run_history": run_records[-20:],
    }


def update_summary(output_dir, report):
    summary_path = output_dir / "immutable-rules-summary.json"
    entries = []
    if summary_path.exists():
        try:
            with open(summary_path, encoding='utf-8') as f:
                data = json.load(f)
                entries = data.get("entries", [])
        except Exception:
            entries = []

    entries.append({
        "generated_at": report["report_metadata"]["generated_at"],
        "overall_status": report["summary"]["overall_status"],
        "pass_rate_pct": report["summary"]["pass_rate_pct"],
        "total_runs": report["summary"]["total_runs_analyzed"],
        "total_failures": report["summary"]["total_failures"],
        "cron_drift_seconds": report["report_metadata"]["cron_drift_seconds"],
        "lookback_days": report["report_metadata"]["lookback_days"],
    })
    entries = entries[-MAX_SUMMARY_RUNS:]

    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump({
            "_note": ("Immutable Rules Gate compliance summary. "
                      "Full reports are GitHub Actions artifacts (90d retention)."),
            "last_updated": report["report_metadata"]["generated_at"],
            "entries": entries,
        }, f, indent=2)
    print(f"Updated summary: {summary_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--output-dir", default="tests/reports")
    p.add_argument("--actual-time", default=None)
    p.add_argument("--expected-time", default=None)
    p.add_argument("--drift-seconds", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    actual_time = args.actual_time or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    expected_time = args.expected_time or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    since = now - timedelta(days=args.lookback_days)

    print(f"Repo: {args.repo} | Lookback: {args.lookback_days}d | Since: {since.strftime('%Y-%m-%d')}")
    client = GitHubClient(token=args.token, repo=args.repo)

    workflow_id = client.get_workflow_id()
    if workflow_id is None:
        print(f"WARNING: '{WORKFLOW_FILENAME}' not found — generating empty report.")
        runs = []
    else:
        print(f"Workflow ID: {workflow_id}")
        runs = client.get_workflow_runs(workflow_id, since)
        print(f"Runs found: {len(runs)}")

    report = build_report(
        runs=runs, client=client,
        actual_time=actual_time, expected_time=expected_time,
        drift_seconds=args.drift_seconds,
        lookback_days=args.lookback_days, repo=args.repo,
    )

    ts = actual_time.replace(":", "").replace("-", "")[:15]
    full_path = output_dir / f"immutable-rules-report-{ts}.json"
    latest_path = output_dir / "immutable-rules-report-latest.json"

    for path in (full_path, latest_path):
        with open(path, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    print(f"Report: {full_path}")

    update_summary(output_dir, report)

    s = report["summary"]
    sep = "=" * 60
    print(f"\n{sep}\nIMMUTABLE RULES GATE — COMPLIANCE REPORT\n{sep}")
    print(f"Period:     last {args.lookback_days} days")
    print(f"Runs:       {s['total_runs_analyzed']} | Pass: {s['total_pass']} | Fail: {s['total_failures']}")
    print(f"Pass rate:  {s['pass_rate_pct']}%")
    print(f"Overall:    {s['overall_status']}")
    print(f"Cron drift: {args.drift_seconds}s ({'OK' if abs(args.drift_seconds) <= 3600 else 'WARNING'})")
    print(sep)
    for rule in report["rule_breakdown"]:
        icon = "✓" if rule["fail"] == 0 else "✗"
        print(f"  {icon} {rule['name']}: {rule['pass_rate_pct']}% ({rule['fail']} failures)")
    if report["recent_failures"]:
        print(f"\nRecent failures ({len(report['recent_failures'])}):")
        for fail in report["recent_failures"]:
            print(f"  - {fail['created_at']} | {fail['run_url']}")
            for r in fail["failed_rules"]:
                print(f"      ✗ {RULE_JOBS.get(r, r)}")

    if s["overall_status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
