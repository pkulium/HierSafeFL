# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated messagermation back to clients
import copy
import aggregator 
import torch
import random
from phe import paillier
import numpy as np
import math
import time
epsilon = 1e-9

class Cloud():

    def __init__(self, shared_layers, device, args):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.device = device
        if args.model == 'lenet':
            self.init_state = torch.flatten(shared_layers.fc2.weight)
        elif args.model == 'cnn_complex':
            self.init_state = torch.flatten(shared_layers.fc_layer[-1].weight)
        elif args.model == 'resnet18':
            self.init_state = torch.flatten(shared_layers.linear.weight)
        self.clock = []
        self.client_client_similarity = None
        self.num_reference = args.num_reference
        self.reference = self.get_reference()
        self.parameter_count = 0
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.a = None
        self.s = None
        start_time = time.time()
        self.s_prime = self.get_s_prime()
        end_time = time.time()
        print(f'time for encryption is {end_time - start_time}')
        self.client_reputation = None
        self.client_learning_rate = None
        self.edge_learning_rate = None
        self.client_comit = {}
        self.edge_comit = {} 


    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        edge.reference = self.reference.to(self.device)
        return None

    def receive_from_edge(self, message):
        edge_id = message['id']
        self.receiver_buffer[edge_id] = {'eshared_state_dict': message['eshared_state_dict'],
                                        'client_reference_similarity':message['client_reference_similarity'],
                                        'comit': message['comit'],
                                    }
        return None

    def foolsgold(self, similarity_client_referecence):
        similarity_client_referecence = {k:v for tmp in similarity_client_referecence for k,v in tmp.items()}
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)
        
        n = len(similarity_client_referecence)
        cs = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cs[i][j] = cos(similarity_client_referecence[i], similarity_client_referecence[j]).item()
        #  Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99
        
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        return cs, wv

    def contra(self, similarity_client_referecence):
        similarity_client_referecence_ = similarity_client_referecence
        similarity_client_referecence = {k:v for _, tmp in similarity_client_referecence.items() for k,v in tmp.items()}
        if self.client_reputation is None:
            self.client_reputation = torch.ones((len(similarity_client_referecence)))
        if self.client_learning_rate is None:
            self.client_learning_rate = torch.ones(len(similarity_client_referecence)) 

        n = len(similarity_client_referecence)
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)
        tao = torch.zeros(n)
        topk = n // 5
        t = 0.5
        delta = 0.1

        cs = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cs[i][j] = cos(similarity_client_referecence[i], similarity_client_referecence[j]).item()

        maxcs = torch.max(cs, dim = 1).values + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        for i in range(n):
            tao[i] = torch.mean(torch.topk(cs[i], topk).values)
            if tao[i] > t:
                self.client_reputation[i] -= delta 
            else:
                self.client_reputation[i] += delta 

        #  Pardoning: reweight by the max value seen
        self.client_learning_rate = torch.ones((n)) - tao
        self.client_learning_rate /= torch.max(self.client_learning_rate)
        self.client_learning_rate[self.client_learning_rate==1] = 0.99
        self.client_learning_rate = (torch.log((self.client_learning_rate / (1 - self.client_learning_rate)) + epsilon) + 0.5)
        self.client_learning_rate[(torch.isinf(self.client_learning_rate) + self.client_learning_rate > 1)] = 1
        self.client_learning_rate[(self.client_learning_rate < 0)] = 0
        self.client_learning_rate /= torch.sum(self.client_learning_rate)
        self.client_reputation /= torch.max(self.client_reputation)
        self.edge_learning_rate = {id: sum([self.client_learning_rate[k] for k in tmp.keys()]) for id, tmp in similarity_client_referecence_.items()}
        print(self.client_learning_rate)
        print(self.edge_learning_rate)
        return cs, self.client_reputation

    def aggregate(self, args):
        similarity_client_reference = {id:dict['client_reference_similarity'] for id, dict in self.receiver_buffer.items()}
        self.client_client_similarity, self.client_reputation = self.contra(similarity_client_reference)
   
        eshared_state_dict = {id:dict['eshared_state_dict'] for id, dict in self.receiver_buffer.items()}
        self.shared_state_dict = aggregator.average_weights_contra_cloud(w=eshared_state_dict, lr = self.edge_learning_rate)
        return None

    def get_client_repuation(self, edge):
        client_repuation = {}
        for id in edge.cids:
            client_repuation[id] = self.client_reputation[id]
        return client_repuation

    def get_client_learning_rate(self, edge):
        client_learning_rate =  {}
        for id in edge.cids:
            client_learning_rate[id] = self.client_learning_rate[id]
        return client_learning_rate

    def send_to_edge(self, edge):
        client_reputation = self.get_client_repuation(edge)
        client_learning_rate = self.get_client_learning_rate(edge)
        message = {
            'shared_state_dict': self.shared_state_dict,
            'client_reputation': client_reputation,
            'client_learning_rate': client_learning_rate,
        }
        edge.receive_from_cloudserver(message)
        return None

    def get_reference(self):
        self.parameter_count = self.init_state.size()[0]
        self.parameter_count = int(self.parameter_count)
        if self.num_reference == 0:
            reference = torch.eye(self.parameter_count, device=self.device)
            return reference
        nonzero_per_reference =  self.parameter_count // self.num_reference
        reference = torch.zeros((self.num_reference,  self.parameter_count), device=self.device)
        parameter_index_random = list(range( self.parameter_count))
        random.shuffle(parameter_index_random)

        for reference_index in range(self.num_reference):
            index = parameter_index_random[reference_index * nonzero_per_reference: (reference_index + 1) * nonzero_per_reference]
            index = torch.tensor(index)
            reference[reference_index][index] = 1
        return reference
    
    def get_s_prime(self):
        # self.a =  torch.tensor(random.sample(range(0, 10000), self.reference.shape[0]), dtype=torch.float32, device=self.device)
        self.a =  torch.rand(self.reference.shape[0], dtype=torch.float32, device=self.device)
        self.s = torch.matmul(self.a, self.reference)
        # s_prime = [self.public_key.encrypt(s_) for s_ in self.s.tolist()]
        s_prime = (self.s + 1) % 7
        return s_prime
        
    def client_register(self, client):
        self.send_to_client(client)
        return

    def receive_from_client(self, message):
        client_id = message['client_id']
        comit  = message['comit']
        self.client_comit[client_id] = comit
        return

    def comit(self, grad):
        ret = []
        for i in range(len(self.s_prime)):
            ret.append(self.s_prime[i] * float(grad[i]))
        
        for i in range(1, len(self.s_prime)):
            ret[0] += ret[i]

        return ret[0] / float(torch.norm(grad)), torch.norm(grad)

    def verify_grad(self, edge_id, cid):
        self.edge_comit[edge_id] =  self.comit(self.receiver_buffer[edge_id]['comit'])
        left = self.edge_comit[edge_id][0] * float(self.edge_comit[edge_id][1])
        ret = []
        for id in cid:
            ret.append(self.client_comit[id][0] * float(self.client_comit[id][1]))
        for i in range(1, len(ret)):
            ret[0] += ret[i]
        left = self.private_key.decrypt(left)
        right = self.private_key.decrypt(ret[0])
        return math.isclose(left, right)

    def verify_cos(self, edge_id):
        client_reference_similarity = self.receiver_buffer[edge_id]['client_reference_similarity']
        for id in client_reference_similarity:
            similarity = client_reference_similarity[id]
            left = self.client_comit[id][0]
            right = (torch.norm(self.reference, dim = 1) * self.a).dot(similarity)

            left = self.private_key.decrypt(left)
            right = float(right)
            if not math.isclose(left, right):
                # return False
                continue
        return True

    def send_to_client(self, client):
        client_id = client.id
        if client_id not in self.client_comit:
            message = {
                's_prime': self.s_prime,
                'private_key': self.private_key,
            }
            self.client_comit[client_id] = None 
        else:
            message = {
                'receipt': 'receipt',
            }
        client.receive_from_cloud(message)