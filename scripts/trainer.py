import torch
from tqdm import tqdm 

class Trainer:
    def __init__(self, model, opt, criterion, train_dataloader, test_dataloader, out_dir, epochs, device):
        self.model = model
        self.opt = opt
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.out_dir = out_dir
        self.epochs = epochs
        self.device = device

    def fit(self):
        with open(f'{self.out_dir}/logs.csv', 'w') as file:
            file.write('epoch,loss,accuracy,best_accuracy,val_loss,val_accuracy,best_val_accuracy\n')
        
        best_accuracy = 0.0 
        best_eval_accuracy = 0.0
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train() 
            
            running_loss = 0.0  
            running_acc = 0.0
            
            running_eval_loss = 0.0 
            running_eval_acc = 0.0
        
            for x,y in iter(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.opt.zero_grad() 
                
                
                pred = self.model(x) 
                loss = self.criterion(pred,y) 
                loss.backward() 
                self.opt.step() 
                
                running_loss += loss.detach() 
                
                accuracy = torch.div(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)),
                                     y.size(dim=0)) 
                
                running_acc += accuracy 
                    
            running_loss = running_loss/len(self.train_dataloader) 
            running_acc = running_acc/len(self.train_dataloader)
        
            if running_acc > best_accuracy: 
                    best_accuracy = running_acc
            
            self.model.eval()
            for x,y in iter(self.test_dataloader): 
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(x) 
                loss = self.criterion(pred,y) 
                
                running_eval_loss += loss.detach()
                
                accuracy = torch.div(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)),
                                     y.size(dim=0)) 
                running_eval_acc += accuracy
            
            running_eval_loss = running_eval_loss/len(self.test_dataloader)
            running_eval_acc = running_eval_acc/len(self.test_dataloader)
        
            if running_eval_acc>= best_eval_accuracy:
                    best_eval_accuracy = running_eval_acc
                    torch.save(self.model, f'{self.out_dir}/best.pt') 
        
            
            print(f'EPOCH {epoch+1}: Loss: {running_loss}, Accuracy: {running_acc}, Best_accuracy: {best_accuracy}')
            print(f'Val_loss: {running_eval_loss}, Val_accuracy: {running_eval_acc}, Best_val_accuracy: {best_eval_accuracy}')
            print()
                
            with open(f'{self.out_dir}/logs.csv', 'a') as file:
                file.write(f'{epoch+1},{running_loss},{running_acc},{best_accuracy},{running_eval_loss},{running_eval_acc},{best_eval_accuracy}\n')
                  
                    
        torch.save(self.model, f'{self.out_dir}/last.pt') 
                
    