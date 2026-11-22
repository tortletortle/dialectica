"""Small synthetic demonstration; no real LLM or factual claims."""
from .engine import AgentProfile, Task, ThompsonScheduler

def main():
    agents = [AgentProfile("Socratic"), AgentProfile("Risk"), AgentProfile("Builder")]
    scheduler = ThompsonScheduler(agents, seed=42, exploration_floor=.15)
    # Synthetic ground-truth success probabilities used only to demonstrate learning.
    truth = {"Socratic": .75, "Risk": .65, "Builder": .45}
    task = Task("demo-001", "architecture", impact=1, uncertainty=.8, conflict=.7)
    for _ in range(120):
        pick = scheduler.select(task)
        success = scheduler.rng.random() < truth[pick.agent]
        scheduler.observe(pick.agent, task.domain, success)
    print("Synthetic posterior means:")
    for name, domains in scheduler.snapshot().items():
        print(name, domains.get("architecture", {}))

if __name__ == "__main__":
    main()
