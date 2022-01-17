from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, input_len, target_len):
        self.input = dataframe.input
        self.target = dataframe.target
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.target_len = target_len
        self.pad_token_id = self.tokenizer.pad_token_id 
        
        
    def __len__(self):
        return len(self.input)

    
    def __getitem__(self, index):
        input_text = str(self.input[index])
        target_text = str(self.target[index])
        input_ = self.tokenizer.batch_encode_plus([input_text], max_length=self.input_len, pad_to_max_length=True, 
                        truncation=True, padding='longest', return_tensors='pt')
        input_ids = input_['input_ids'] 
        input_mask = input_['attention_mask'] 
        target_ = self.tokenizer.batch_encode_plus([target_text], max_length=self.target_len, pad_to_max_length=True, 
                        truncation=True, padding='longest', return_tensors='pt')
        labels = target_['input_ids'] 
        batch = {'input_ids': input_ids, 'attention_mask': input_mask, 'labels': labels}
        return batch
    
    
    def collate_fn(self, batch):
        input_ids = pad_sequence([x["input_ids"][0] for x in batch], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([x["attention_mask"][0] for x in batch], batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence([x["labels"][0] for x in batch], batch_first=True, padding_value=self.pad_token_id)
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
