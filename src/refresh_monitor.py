# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorRefreshInfoState
import time

dbutils.widgets.text("catalog", "fraud_demo_test")
catalog = dbutils.widgets.get("catalog")

table_name = f"{catalog}.serving.fraud_detector_payload"

w = WorkspaceClient()

print(f"Triggering refresh for monitor on {table_name}")
refresh = w.quality_monitors.run_refresh(table_name=table_name)
refresh_id = refresh.refresh_id
print(f"Refresh started: {refresh_id}")

TERMINAL_STATES = (
    MonitorRefreshInfoState.SUCCESS,
    MonitorRefreshInfoState.FAILED,
    MonitorRefreshInfoState.CANCELED,
)

max_wait_seconds = 600
waited = 0
status = None
while waited < max_wait_seconds:
    status = w.quality_monitors.get_refresh(
        table_name=table_name, refresh_id=refresh_id)
    print(f"Refresh state: {status.state}")
    if status.state in TERMINAL_STATES:
        break
    time.sleep(15)
    waited += 15

if status is None or status.state != MonitorRefreshInfoState.SUCCESS:
    raise RuntimeError(
        f"Monitor refresh did not succeed: {status.state if status else 'timed out'}")

print("Monitor refresh completed successfully.")
