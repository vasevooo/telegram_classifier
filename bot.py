import torch
import re
import numpy as np
import string
from collections import Counter
from nltk.corpus import stopwords
import json
from aiogram import Bot, Dispatcher, executor, types
from model import RNNNet
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the model
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
N_LAYERS = 2
VOCAB_SIZE = 222610
SEQ_LEN = 146

model = RNNNet(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM, emb_size=EMBEDDING_DIM, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
model.load_state_dict(torch.load('models/10_monday_best.pt', map_location=torch.device('cpu')))


# Preprocess functions

def data_preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub('<.*?>', '', text)  # html tags
    text = ''.join([c for c in text if c not in string.punctuation])  # Remove punctuation
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text


def padding(review_int: list, seq_len: int) -> np.array:
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[:seq_len]
        features[i, :] = np.array(new)

    return features


def preprocess_comment(input_string: str, seq_len: int, vocab_to_int_path: str) -> list:
    with open('models/imdb.json', 'r') as file:
        vocab_to_int = json.load(file)

    preprocessed_string = data_preprocessing(input_string)
    result_list = []
    for word in preprocessed_string.split():
        try:
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            print(f'{e}: not in dictionary!')

    preprocessed_comment = padding([result_list], seq_len)[0]

    return torch.tensor(preprocessed_comment)


# Prediction function
def predict_sentiment(comment):
    preprocessed_comment = preprocess_comment(comment, SEQ_LEN, 'models/imdb.json')

    model.eval()
    result = model(preprocessed_comment.unsqueeze(0)).sigmoid().round().item()
    # Return the predicted sentiment
    return result


# Telegram bot handler
@dp.message_handler(commands=['start'])
async def handle_message(message: types.Message):
    user_comment = message.text
    user_name = message.from_user.full_name
    reply_text = f'{user_name}, please enter your comment in English:'
    sentiment = predict_sentiment(user_comment)
    response = "Positive" if sentiment == 1 else "Negative"
    await message.reply(response)






bot = Bot(token='5860638077:AAF-JQCksGqtBA4YRQ5u60sNCMGUWlynEXQ')
dp = Dispatcher(bot)
dp.register_message_handler(handle_message)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

