from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class EnterpriseTask:
    employee_name: str
    department: str
    request_type: str
    amount: float

    approved: bool = False
    rejected: bool = False
    execution_status: str = "NOT_STARTED"

    logs: List[str] = field(default_factory=list)

    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.logs.append(log_message)
        print(log_message)


class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def process(self, task: EnterpriseTask) -> bool:
        raise NotImplementedError


class ValidationAgent(BaseAgent):
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Starting validation.")
        if task.amount <= 0:
            task.rejected = True
            task.add_log(f"{self.name}: Invalid amount.")
            return False
        if task.request_type == "":
            task.rejected = True
            task.add_log(f"{self.name}: Missing request type.")
            return False
        task.add_log(f"{self.name}: Validation successful.")
        return True


class RiskAssessmentAgent(BaseAgent):
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Performing risk assessment.")
        if task.amount > 50000:
            task.rejected = True
            task.add_log(f"{self.name}: Request exceeds risk threshold.")
            return False
        task.add_log(f"{self.name}: Risk assessment passed.")
        return True


class ComplianceAgent(BaseAgent):
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Running compliance checks.")
        restricted_departments = ["Blacklisted Division"]
        if task.department in restricted_departments:
            task.rejected = True
            task.add_log(f"{self.name}: Department restricted by policy.")
            return False
        task.add_log(f"{self.name}: Compliance checks passed.")
        return True


class ApprovalAgent(BaseAgent):
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Manager approval started.")
        if task.amount < 10000:
            task.approved = True
            task.add_log(f"{self.name}: Automatically approved.")
            return True
        elif task.amount <= 50000:
            task.approved = True
            task.add_log(f"{self.name}: Approved after management review.")
            return True
        else:
            task.rejected = True
            task.add_log(f"{self.name}: Approval rejected.")
            return False


class ExecutionAgent(BaseAgent):
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Starting execution.")
        if not task.approved:
            task.execution_status = "FAILED"
            task.add_log(f"{self.name}: Cannot execute unapproved request.")
            return False
        task.execution_status = "COMPLETED"
        task.add_log(f"{self.name}: Task executed successfully.")
        return True


class AuditAgent(BaseAgent):
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Recording audit information.")
        task.add_log(f"{self.name}: Final status = {task.execution_status}")
        return True


class NotificationAgent(BaseAgent):
    """Simple notification agent (simulation)."""
    def process(self, task: EnterpriseTask) -> bool:
        task.add_log(f"{self.name}: Sending notification - Approved={task.approved}, Rejected={task.rejected}")
        return True


class MultiAgentOrchestrator:
    def __init__(self):
        self.registry: Dict[str, BaseAgent] = {}

    def register(self, name: str, agent: BaseAgent):
        self.registry[name] = agent

    def run_sequence(self, task: EnterpriseTask, sequence: List[str]) -> bool:
        for name in sequence:
            agent = self.registry.get(name)
            if agent is None:
                task.add_log(f"Orchestrator: Agent '{name}' not registered. Skipping.")
                continue
            success = agent.process(task)
            if not success:
                task.add_log(f"Orchestrator: Stopped at {name} (failed).")
                return False
        return True


if __name__ == "__main__":
    task = EnterpriseTask(
        employee_name="Ahmed Hassan",
        department="Finance",
        request_type="Cloud Infrastructure Purchase",
        amount=15000
    )

    orch = MultiAgentOrchestrator()
    orch.register("validation", ValidationAgent("Validation Agent"))
    orch.register("risk", RiskAssessmentAgent("Risk Assessment Agent"))
    orch.register("compliance", ComplianceAgent("Compliance Agent"))
    orch.register("approval", ApprovalAgent("Approval Agent"))
    orch.register("execution", ExecutionAgent("Execution Agent"))
    orch.register("audit", AuditAgent("Audit Agent"))
    orch.register("notify", NotificationAgent("Notification Agent"))

    sequence = ["validation", "risk", "compliance", "approval", "execution", "audit", "notify"]

    print("\n===============================================")
    print("ENTERPRISE WORKFLOW STARTED")
    print("===============================================")

    success = orch.run_sequence(task, sequence)

    if success:
        print("\nWorkflow completed successfully.")
    else:
        print("\nWorkflow stopped due to failure.")

    print("\n================================================")
    print("FINAL ENTERPRISE REPORT")
    print("================================================")
    print(f"Employee Name : {task.employee_name}")
    print(f"Department    : {task.department}")
    print(f"Request Type  : {task.request_type}")
    print(f"Amount        : {task.amount}")
    print(f"Approved      : {task.approved}")
    print(f"Rejected      : {task.rejected}")
    print(f"Execution     : {task.execution_status}")
    print("================================================")

    print("\n================================================")
    print("AUDIT LOGS")
    print("================================================")
    for log in task.logs:
        print(log)
