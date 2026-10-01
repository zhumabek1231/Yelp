from fastapi import FastAPI
import uvicorn
import torch
import torch.nn as nn

from torchtext.data.utils import get_tokenizer


from pydantic import BaseModel

labels = ['1 star', '2 star', '3 stars', '4 stars', '5 stars']

class CheckStars(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 64)
    self.lstm = nn.LSTM(64, 128, batch_first=True)
    self.fc = nn.Linear(128, 5)

  def forward(self,x):

    x = self.embedding(x)

    _, (hidden_state, _) = self.lstm(x)

    x = hidden_state.squeeze(0)
    x = self.fc(x)

    return x

yelp_app = FastAPI()
vocab = torch.load('vocab_yelp1.pth', map_location='cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckStars(len(vocab)).to(device)
model.load_state_dict(torch.load('yelp1.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = get_tokenizer('basic_english')
def replase_words(name):
  return [vocab[i] for i in tokenizer(name)]


class TextShema(BaseModel):
    word:str


@yelp_app.post('/predict')
async def change_text(text:TextShema):
    num_text = torch.tensor(replase_words(text.word)).unsqueeze(0).to(device)


    with torch.no_grad():
        pred = model(num_text)
        result = torch.argmax(pred,dim=-1).item()
        return {'class':labels[result]}






if __name__=='__main__':
    uvicorn.run(yelp_app,host='127.0.0.1',port=8000)