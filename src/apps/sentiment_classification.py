from datasets import load_dataset
import torch
import cvxpy as cp
from transformers import AutoTokenizer, AutoModel

import torch.nn as nn
import torch.nn.functional as F


device = 'cuda:0'

model = AutoModel.from_pretrained(
        "microsoft/phi-2", 
        device_map = device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        ).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token
print("microsoft/phi-2")


torch.random.manual_seed(seed=42)
N = 50; N_test = 20
imdb = load_dataset("imdb")
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(N))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(N_test))])
ytrain = (2*torch.Tensor([i['label'] for i in small_train_dataset])-1).to(device)
ytest = (2*torch.Tensor([i['label'] for i in small_test_dataset])-1).to(device)


def preprocess_function(examples):
   return tokenizer(examples["text"], return_tensors='pt', padding=True, truncation=True)

imdb_train_input_ids_ =  small_train_dataset.map(preprocess_function, batched=True)
imdb_train_input_ids = torch.Tensor([i['input_ids'] for i in imdb_train_input_ids_]).long().to(device)
imdb_train_attention_mask = torch.Tensor([i['attention_mask'] for i in imdb_train_input_ids_]).to(device)
imdb_test_input_ids_ = small_test_dataset.map(preprocess_function, batched=True)
imdb_test_input_ids = torch.Tensor([i['input_ids'] for i in imdb_test_input_ids_]).long().to(device)
imdb_test_attention_mask = torch.Tensor([i['attention_mask'] for i in imdb_test_input_ids_]).to(device)

with torch.no_grad():
    imdb_train_outputs = model(imdb_train_input_ids)
    imdb_test_outputs = model(imdb_test_input_ids)

imdb_train_word_embeddings = imdb_train_outputs.last_hidden_state.to(device)
imdb_test_word_embeddings = imdb_test_outputs.last_hidden_state.to(device)

imdb_train_masked_word_embeddings = imdb_train_word_embeddings * imdb_train_attention_mask.unsqueeze(-1).float() 
imdb_train_sentence_embeddings = imdb_train_masked_word_embeddings[torch.arange(N),(torch.count_nonzero(imdb_train_attention_mask,dim=1)-2).long(),:]

imdb_test_masked_word_embeddings = imdb_test_word_embeddings * imdb_test_attention_mask.unsqueeze(-1).float() 
imdb_test_sentence_embeddings = imdb_test_masked_word_embeddings[torch.arange(N_test),(torch.count_nonzero(imdb_test_attention_mask,dim=1)-2).long(),:]


Xtrain = imdb_train_sentence_embeddings[:,0:200]
Xtest = imdb_test_sentence_embeddings[:,0:200]


Xtrain = Xtrain[:,0:200]
Xtest = Xtest[:,0:200]

print(f'{Xtest.shape=},{Xtest.device=}')

"""create convex NN"""
def relu(x):
    return torch.maximum(0,x)
def drelu(x):
    return x>=0
R = 1
N = Xtrain.shape[0]
d = Xtrain.shape[1]
P = 500
dmat=torch.empty((N,0)).to(device)
for i in range(P):
    u=torch.randn(d,1).to(device)
    dmat = torch.hstack([dmat,drelu(Xtrain@u)])
dmat=(torch.unique(dmat,dim=1))
m=dmat.shape[1]
print(f'number of plane used: {m}')


"""create two-layer NN"""
learning_rate = 1e-2
print(f'{learning_rate=}')
class Net(nn.Module):
    def __init__(self,input_dim,hidden_neuron):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(input_dim, hidden_neuron)
      self.fc2 = nn.Linear(hidden_neuron, 1)
    def forward(self, x):
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       return x
twolayer_nn = Net(input_dim=d,hidden_neuron=2*m).to(device)



def center(S, R=1, boxinit=False, step_size = 1e-2):
    print('begin center')
    s = cp.Variable(2*d*m)
    obj = 0 if boxinit else cp.log(R - cp.norm(s))
    constraints = []
    if len(S)>0:
        obj += cp.sum([cp.log(rhs - lhs @ s) for lhs, rhs in S])
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK)
    print('end center')
    return s.value
   

def pred_point_simplified_vec(i, vec):
    return (dmat[i] @ torch.kron(torch.eye(len(dmat[i])).to(device), torch.concatenate((Xtrain[i], -Xtrain[i])).T).to(device)) @ vec

def cut(S, x, y, dmat_row):
    m = len(dmat_row)
    S.append(((-y * dmat_row @ torch.kron(torch.eye(m).to(device), torch.concatenate((x, -x)).T)).cpu().detach().numpy(), 0))
    
    relu_constraint = -torch.kron(torch.diag(2*dmat_row-torch.ones(m).to(device)), torch.kron(torch.eye(2).to(device), x)).cpu().detach().numpy()
    for lhs in relu_constraint:
        S.append((lhs , 0))

def query(c, data_tried, data_used, M=100, iter=0):
    mini = torch.inf
    i_mini = -1
    maxi = -torch.inf
    i_maxi = -1 
    minabs = torch.inf
    i_minabs = -1

    for i in range(N): 
        if i not in data_tried and i not in data_used:
            pred = pred_point_simplified_vec(i, c)
            if pred < mini:
                i_mini = 1*i
                mini = pred
            if pred > maxi:
                i_maxi = 1*i
                maxi = pred
            if abs(pred) < minabs:
                i_minabs = 1*i
                minabs = abs(pred)
    return i_mini, i_maxi, i_minabs


def cutting_plane(n_points=100, maxit=100, boxinit=False):
    score = []
    data_tried = []
    data_used = []
    Ct = []
    c = None
    did_cut = True
    it = 0
    while len(data_used) < n_points and it < maxit:
        if len(data_tried) == N:
            data_tried = []
        if did_cut:
            try:
                c = torch.Tensor(center(Ct, R=R)).to(device)
            except:
                print('solver ERROR')
                pass
            did_cut = False
        i_mini, i_maxi, i_minabs = query(c, data_tried, data_used,iter=it)
        data_tried += [i_minabs] 
        data_tried = list(set(data_tried))
        theta_matrix = c.reshape((2*d, m))
        v1_list = theta_matrix[:d]; v2_list = theta_matrix[d:]
        ytest_cvx=torch.sign(torch.sum(drelu(Xtest@v1_list)*(Xtest@v1_list)-drelu(Xtest@v2_list)*(Xtest@v2_list),axis=1)).to(device)
        ytrain_cvx=torch.sign(torch.sum(drelu(Xtrain@v1_list)*(Xtrain@v1_list)-drelu(Xtrain@v2_list)*(Xtrain@v2_list),axis=1)).to(device)
        print(f'iter: {it}','convex test accuracy: ', torch.sum(ytest_cvx == ytest) / len(ytest),' #points: ', len(data_used))
        ytest_twolayer = ((torch.sigmoid(twolayer_nn(Xtest)))>0.5).int()*2-1
        print(f'iter: {it}','twolayer test accuracy: ', torch.sum(ytest_twolayer.squeeze() == ytest) / len(ytest))
        if len(score)==0:
            score.append((torch.sum(ytest_cvx == ytest) / len(ytest), len(data_used)))
        else:
            if len(data_used)==score[-1][1]:
                pass 
            else:
                score.append((torch.sum(ytest_cvx == ytest) / len(ytest), len(data_used)))
        if True: 
            print(f'{i_minabs=},{len(data_tried)=},{len(data_used)=}')
            if torch.sign(pred_point_simplified_vec(i_minabs, c)) != ytrain[i_minabs]:
                if i_minabs not in data_used:
                    # convex cut 
                    print(f'Cutting at iteration {it}, index {i_minabs}')
                    cut(Ct, Xtrain[i_minabs], ytrain[i_minabs], dmat[i_minabs])
                    data_used.append(i_minabs)
                    did_cut = True

                    # non-convex update 
                    twolayer_pred = torch.sigmoid(twolayer_nn(Xtrain[i_minabs]))
                    y_revise = (ytrain[i_minabs]+1)/2
                    loss = -(y_revise*torch.log(twolayer_pred)+(1-y_revise)*torch.log(1-twolayer_pred))
                    twolayer_nn.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for name, param in twolayer_nn.named_parameters():
                            if param.requires_grad:
                                param.data -= learning_rate * param.grad


        

        it += 1
    
    return Ct, c, data_used, score

C, c, used, score = cutting_plane(50)





   
