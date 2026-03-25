from config import MAX_STEPS

def run_agent(agent, question):
    trajectory = []
    history = ""

    for step_id in range(MAX_STEPS):
        step = agent.step(question, history)

        trajectory.append({
            "step_id": step_id,
            **step
        })

        history += f"\nStep {step_id}: {step}"

    return trajectory