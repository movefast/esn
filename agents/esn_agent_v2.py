import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import agent
# from replay_buffer_episodic import ReplayMemory, Transition
from agents.replay.replay_buffer import ReplayMemory, Transition

criterion = torch.nn.SmoothL1Loss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.tanh = nn.ReLU()
#         self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
#         for param in self.i2h.parameters():
#             param.requires_grad = False
#         u,s,v = torch.svd(self.i2h.weight)
#         with torch.no_grad():
#             self.i2h.weight /= s[0].item()
#         self.i2o = nn.Linear(input_size+hidden_size, output_size)

#     def forward(self, x, hidden):
#         x = torch.cat([x,hidden],dim=1)
#         x = self.tanh(x)
#         # get new hidden
#         h = self.i2h(x)
#         h = self.tanh(h)

#         x = self.i2o(x)
#         return x

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size).to(device)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.i2h = nn.Linear(input_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.i2h.weight)
        # torch.nn.init.uniform_(self.i2h.weight, -1, 1)
        # self.i2h.weight.requires_grad_(False)

        self.w = nn.Linear(hidden_size, hidden_size)
        # self.w = nn.Conv1d(1, 1, kernel_size=5, stride=2, dilation=2, padding=1)
        # for param in self.i2h.parameters():
        #     param.requires_grad = False
        torch.nn.init.uniform_(self.w.weight, -0.5, 0.5)
        with torch.no_grad():
            self.w.weight[torch.rand_like(self.w.weight) < 0.5] = 0
            u,s,v = torch.svd(self.w.weight)
            self.w.weight /= s[0].item()
        self.w.weight.requires_grad_(False)
        # self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.i2o = nn.Linear(input_size+hidden_size, (input_size+hidden_size)//2)
        self.o = nn.Linear((input_size+hidden_size)//2, output_size)
        torch.nn.init.xavier_uniform_(self.i2o.weight)
        # self.actions = nn.Parameter(torch.normal(0, .01, (output_size, hidden_size)))

    def forward(self, inp, hidden):
        output = []
        hiddens = []
        if len(inp.size()) == 2:
            inp = inp.unsqueeze(1)
        for i in range(inp.size(0)):
            x = inp[i]

            with torch.no_grad():
                hidden_w = self.w(hidden)

                # hidden_w = self.w(hidden.unsqueeze(0))
                # hidden_w.squeeze_(0)
            hidden_in = self.i2h(x)

            hidden = self.tanh(hidden_in+hidden_w)

            combined = torch.cat((x, hidden), 1)

            # output.append(self.i2o(combined))
            output.append(self.o(self.tanh(self.i2o(combined))))
            hiddens.append(hidden)

        return output[-1], hiddens[-1]

    def batch(self, inp, hidden, discount_batch, action_batch):
        output = []
        hiddens = []
        if len(inp.size()) == 2:
            inp = inp.unsqueeze(1)
        for i in range(inp.size(0)):
            x = inp[i]

            with torch.no_grad():
                hidden_w = self.w(hidden)

                # hidden_w = self.w(hidden.unsqueeze(0))
                # hidden_w.squeeze_(0)
            hidden_in = self.i2h(x)
            hidden = self.tanh(hidden_in+hidden_w)

            combined = torch.cat((x, hidden), 1)

            # output.append(self.i2o(combined))
            output.append(self.o(self.tanh(self.i2o(combined))))
            # hidden = hidden * (1 + self.actions[action_batch[i]])

            hiddens.append(hidden.detach())

            if discount_batch[i].item() == 0:
                hidden = self.initHidden()
        return torch.cat(output), hiddens

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)


class RNNAgent(agent.BaseAgent):
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        torch.manual_seed(agent_init_info["seed"])
        self.T = agent_init_info.get("T",10)
        self.hidden_size = agent_init_info.get("hidden_size", self.num_states+1)

        self.rnn = SimpleRNN(self.num_states+1, self.hidden_size, self.num_actions).to(device)
        self.target_rnn = SimpleRNN(self.num_states+1, self.hidden_size, self.num_actions).to(device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.step_size)
        self.buffer = ReplayMemory(1000)
        self.tau = .5
        self.flag = False
        self.train_steps = 0


    def get_state_feature(self, state):
        state, is_door = state
        state = np.eye(self.num_states)[state]
        state = torch.Tensor(state).to(device)

        # if self.is_door is None or is_door is True:
        #     self.is_door = int(is_door)
        # else:
        #     self.is_door = self.is_door * .9 + is_door * .1

        self.is_door = int(is_door)
        is_door = torch.Tensor([float(self.is_door)]).to(device)
        return torch.cat([state, is_door])[None, ...]

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        # Choose action using epsilon greedy.
        self.is_door = None
        self.feature = None
        state = self.get_state_feature(state)
        self.hidden = self.rnn.initHidden()
        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            current_q = F.softmax(current_q, -1)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        # with torch.no_grad():
        #     self.hidden *= 1 + self.rnn.actions[action]

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps = 0

        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        # Choose action using epsilon greedy.
        state = self.get_state_feature(state)

        self.buffer.push(self.prev_state, self.prev_action, reward, self.hidden.detach(), self.discount)

        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            current_q = F.softmax(current_q, -1)
        current_q.squeeze_()

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        # with torch.no_grad():
        #     self.hidden *= 1 + self.rnn.actions[action]

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        if len(self.buffer) > self.T+1:# and self.steps % 5 == 0:# and self.epsilon == .1:
            self.batch_train()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        state = self.get_state_feature(state)
        if append_buffer:
            self.buffer.push(self.prev_state, self.prev_action, reward, self.hidden.detach(), 0)
            self.flag = True

        if len(self.buffer) > self.T+1:
            self.batch_train()

    def batch_train(self):
        self.train_steps += 1
        self.rnn.train()
        transitions = self.buffer.sample_successive(self.T+1)
        batch = Transition(*zip(*transitions))
        next_discount_batch = torch.FloatTensor(batch.discount[1:]).to(device)
        state_batch = torch.cat(batch.state[:-1])
        next_state_batch = torch.cat(batch.state[1:])

        action_batch = torch.LongTensor(batch.action[:-1]).view(-1, 1).to(device)
        next_action_batch = torch.LongTensor(batch.action[1:]).view(-1, 1).to(device)

        reward_batch = torch.FloatTensor(batch.reward[:-1]).to(device)
        hidden_batch = batch.hidden[0]
        next_hidden_batch = batch.hidden[1] # or after rnn next_hidden_batch[0]

        discount_batch = torch.FloatTensor(batch.discount[:-1]).to(device)

        current_q, _ = self.rnn.batch(state_batch, hidden_batch, discount_batch, action_batch)
        q_learning_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            new_q, _ = self.target_rnn.batch(next_state_batch, next_hidden_batch, next_discount_batch, next_action_batch)
        max_q = new_q.max(1)[0]
        # max_q = new_q.gather(1, next_action_batch).squeeze_()
        target = reward_batch
        target += discount_batch * max_q

        target = target.view(-1, 1)
        # loss = criterion(q_learning_action_values, target)
        loss = criterion(q_learning_action_values[-1], target[-1])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.rnn.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.train_steps % 100 == 0:
            self.update()

    # def update(self):
    #     # target network update
    #     for target_param, param in zip(self.target_rnn.parameters(), self.rnn.parameters()):
    #         target_param.data.copy_(
    #             self.tau * param + (1 - self.tau) * target_param)

    def update(self):
        self.target_rnn.load_state_dict(self.rnn.state_dict())

    def update_target(self):
        self.target_rnn.load_state_dict(self.rnn.state_dict())
