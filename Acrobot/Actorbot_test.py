import gymnasium as gym
from ppo_Actorbot import PPO

TEST_NUM = 100


def test(ppo, env, episodes):
    success = 0
    failure = 0
    final_score = 0
    for i in range(episodes):
        if i == episodes - 1:
            env = gym.make("Acrobot-v1", render_mode="human")
        state = env.reset()[0]
        done = False
        score = 0
        steps = 0
        while not done and steps < 500:
            action = ppo.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            score += reward
            steps += 1
        final_score += steps
        if steps < 500:
            print(f"Episode {(i+1):02d} finished after {steps} timesteps")
            success += 1
        else:
            print(f"Episode {(i+1):02d} did not finish")
            failure += 1
    print(
        f"Success: {success}, Failure: {failure}, Total: {episodes}, Average Steps: {final_score / episodes}"
    )
    print(f"Success Rate: {success / episodes*100:.2f}%")


if __name__ == "__main__":
    model = PPO()
    model.load("Actorbot_actor.pth", "Actorbot_critic.pth")
    env = gym.make("Acrobot-v1")
    test(model, env, TEST_NUM)
