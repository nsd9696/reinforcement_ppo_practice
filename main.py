import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#hyperparameter
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO,self).__init__()
        self.data=[]
        self.gamma = 0.98
        self.lmbda =0.95
        self.eps = 0.1
        self.K = 2

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)

    def pi(self,x,softmax_dim=0): #softmax_dim: Batch처리를 위해 도입 된 친구, 단순한 state에 대한 inference에는 dim=0,
        #state1 ~ state(t) 가 input으로 들어간다면 softmax_dim=1이어야함
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self,x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for item in self.data:
            s, a, r, s_prime, prob_a, done = item

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            #state를 제외한 나머지는 integer형태라서 array형태로 맞춰줌
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), \
                                         torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float), \
                                         torch.tensor(prob_a_lst)
        self.data=[]
        return s, a, r, s_prime, done_mask, prob_a

    def train(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            #GAE(Generalized Advantage Estimation)
            advantage_lst = []
            advantage = 0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype = torch.float)

            #Clipped Loss
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a)) #a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            loss = -torch.min(surr1,surr2) + F.smooth_l1_loss(td_target.detach(), self.v(s))
            #detach(): td_target 앞의 그래프를 모두 떼어냄, 즉 gradient flow가 발생하지 않음
            # =>td_target과 v(s)가 서로 가까워지는 방향으로 loss를 줄여나가는 것이 아니라 td_target은 상수로 고정하고 v(s)만 가까워짐

            #Clipped ratio 설명
            #ratio == 1.3, advantage가 양수일 때 eps == 0.1인 경우 Clip(ratio,1.1,0.9) == 1.1 이다
            #surr1은 surr2보다 크기 때문에 surr1은 loss에 영향을 못 미침
            #surr2에서도 ratio는 clip에 영향을 못미침, 즉 loss에 영향을 못미침
            #즉 clip 범위를 벗어나는 ratio는 loss에 영향을 못미침 => 벗어나면 그 sample은 버리게 됨

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            #PPO는 DQN과 REINFORCE 보다 학습이 좋음, 학습이 빠름


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    gamma = 0.99
    T = 20 # 몇 time step 동안 data를 모을지
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done=False
        while not done:
            for t in range(T): #T스텝만큼 data를 모으고 학습을 함
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s,a,r/100.0, s_prime, prob[a].item(), done))
                #보통은 (s,a,r,s_prime, done)만 저장
                #prob[a]는 실제 한 action의 확률값 => 나중에 ratio 계산할 때 쓰임
                s = s_prime

                score += r
                if done:
                    break
            model.train()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()