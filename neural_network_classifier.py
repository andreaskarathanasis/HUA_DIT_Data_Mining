import torch
from torch import nn

class OscarClassificationNeuralNetwork(nn.Module):
    def __init__(self, input_features: int, hidden_layer_input_features: int, hidden_layer_output_features: int, device: str, output_features: int = 1, learning_rate: float = 0.01):
        super(OscarClassificationNeuralNetwork, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layer_input_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_input_features, out_features=hidden_layer_output_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_output_features, out_features=output_features)
        )

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer= torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train_accuracy_progess = []
        

    # Not using sigmoid here when use BCEWithLogits loss function
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
    
    # Start training model
    def fit(self, accuracy_fn, X_train, y_train, X_test, y_test, epochs, seed=None, verbose=False):
        if seed != None:
            torch.manual_seed(seed)

        device = self.device

        loss_fn=self.loss_fn
        optimizer = self.optimizer       

        # print(f"Training on: {device}")

        epochs = epochs
        
        X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
        
        self.train()

        # for epoch in tqdm(range(epochs), desc='Training...'):
        for epoch in range(epochs):
            y_pred = self(X_train).to(device)

            loss = loss_fn(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate prediction accuracy
            acc = accuracy_fn(y_pred=torch.round(torch.sigmoid(y_pred)), y_true=y_train)

           
            ### Testing
            self.eval()
            with torch.inference_mode():
                y_pred_test = self(X_test)
                test_loss = loss_fn(y_pred_test, y_test)
                test_acc = accuracy_fn(y_pred=torch.round(torch.sigmoid(y_pred_test)), y_true=y_test)
          
            if epoch % int(epochs/10) == 0 or epoch == epochs-1:
                
                self.train_accuracy_progess.append({
                    'epoch': epoch,
                    'train_acc' : acc,
                    'test_acc': test_acc
                })

                if verbose: print(f'Epoch: {epoch:04d} | Loss: {loss:.4f}, Acc: {acc:.2f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
                   
        return self.train_accuracy_progess
    
    
    def test(self, accuracy_fn, X_test, y_test):
        device = self.device
        with torch.inference_mode():
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_test_logits = self(X_test)
            y_test_probs = torch.sigmoid(y_test_logits)
            acc = accuracy_fn(y_pred=torch.round(y_test_probs), y_true=y_test)
        # print(f'Test total accuracy: {acc:.4f}')
        return y_test_logits, acc
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    

# Accuracy calculator helper function
def accuracy_fn(y_pred, y_true):
   
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred))