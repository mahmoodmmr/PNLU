import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

tok = 'media/tokenizer'

class GPT2Persian:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained(tok)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)

    def forward(self, text, max_length=100, num_return_sequences=1):
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences)

        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response_text

class PersianDataset(Dataset):
    def __init__(self, text_file):
        self.text_file = text_file
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        with open(text_file, 'r', encoding='utf-8') as file:
            self.text_list = file.read().splitlines()

        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set the padding token

    def __len__(self):
        return len(self.text_list)


    def __getitem__(self, index):
        text = self.text_list[index]

        # Preprocess and encode the text
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=1024, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask



class PersianModel(nn.Module):
    def __init__(self, model_path):
        super(PersianModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        return logits


def fine_tune_model(dataset, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersianModel(model_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_epochs = 3

    print("epoch...")


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}")
    print("persian_model...")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "persian_model.pth")

def runP():
    # dataset = PersianDataset('media/large_dataset.txt', GPT2Tokenizer.from_pretrained('gpt2'))
    dataset = PersianDataset('media/large_dataset.txt')
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    # Ensure tokenizer and model are using the same vocabulary
    tokenizer.save_pretrained('media/tokenizer')
    model.save_pretrained('media/model')

    print("tunning...")

    # Fine-tune the model
    fine_tune_model(dataset, 'media/model')
    print("tunning done")