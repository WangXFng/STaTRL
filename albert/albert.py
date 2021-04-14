from transformers import AlbertTokenizer, AlbertModel
import torch.nn as nn

# active Service Price
# arts Service Price
# auto Service Price
# restaurants Food Service Price
# food Food Price
# nightlife Food Service Proce
# shopping Service Price
# professional Service Price
# physicians Service Price
# pets Service Price
# hotelstravel Food Service
# health Service Proce
# fitness Service Price
# education Service Price
# beautysvc Service Proce

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# print(tokenizer)
# print(tokenizer.sep_token_id)
model = AlbertModel.from_pretrained("albert-base-v2")
text = "Oh happy day, finally have a Canes near my casa. Yes just as others are griping about the Drive thru " \
       "is packed just like most of the other canes in the area but I like to go sit down to enjoy my chicken. " \
       "The cashiers are pleasant and as far as food wise i have yet to receive any funky chicken. The clean up crew" \
       " zips around the dining area constantly so it's usually well kept. My only gripe is the one fella with Red" \
       " hair he makes the rounds while cleaning but no smile or personality a few nights ago he tossed the napkins" \
       " i just put on the table to help go with my meal. After I was done he just reached for my tray no \"excuse" \
       " me or are you done with that?\"  I realize he's trying to do his job quickly but a little table manners" \
       " goes along way. That being said still like to grub here and glad that there's finally a Cane's close to me."
encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=True)
output = model(**encoded_input)

last_hidden_states = output.last_hidden_state

# active
linear2 = nn.Linear(768, )

# print(output)

