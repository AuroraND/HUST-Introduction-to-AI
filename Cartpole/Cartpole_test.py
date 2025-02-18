import gymnasium as gym
from ppo_Cartpole import PPO

TEST_NUM = 100


def test(ppo, env, episodes):
    final_score = 0
    for i in range(episodes):
        if i == episodes - 1:
            env = gym.make("CartPole-v1", render_mode="human")
        state = env.reset()[0]
        done = False
        score = 0
        while not done and score < 500:
            action = ppo.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            score += reward

        final_score += score
        print(f"Episode {(i+1):02d} gained bonus value of {score}")

    print(f"Total: {episodes}, Average Score: {final_score / episodes}")


if __name__ == "__main__":
    model = PPO()
    model.load("Cartpole_actor.pth", "Cartpole_critic.pth")
    env = gym.make("CartPole-v1")
    test(model, env, TEST_NUM)
