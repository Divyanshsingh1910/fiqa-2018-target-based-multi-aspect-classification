

> Greetings everyone, this is divyansh and I am going to present my work on Barclays IB quant Case study.

> This is are the contents I am going to cover.

> I will begin with my introduction

> I am a final year undergraduate in the computer science department of IIT Kanpur, I am deeply interested in 
recent advancements in the tech and how are these advancement going to impact the future of health, finance and related
sectors. Academically I have pursued ML and systems and majorly their applications. Apart from these I also like to draw
and read blogs online.

> Now, I will move to understanding the problem statement

> So, we have been provided with texts from news and blogs and depending on the targets in the sentence provided 
we need to predict the aspect and sentiment score. Any text can have mulitple targets.

> Let's analyze the data to get more insights about the problem statement 

> If we look distribution of sentiment score 

> New formats doesn't really impact the score 

> label make it too trivial

> preprocess the aspects labels and then see the count 

> aspect classificatin and sentiment score prediction do have correlation between them 

> let's see the key challenge

> not a standard problem of text classification since we have targets. 

> consider this text where we have RIVIAN and TESLA stocks

> depending on the target the aspect and sentiment score changes 

> Now, I will continue with the approach to the problem 

> to give semantic meaning we have first tokenizer then an bert tuned on financial dataset 
the embedding of the cls token givens embedding representation of wholw sentence.
and we call vectorizer module 

> So, now we have vector representation for both the input sentence and target but what about the key challenge of 
target based processing.

> The targets needs attention, let's this is the sentence representation and below is the target representation, Now
from the sentence we need to know sentiment about this target specifically. So, attention heads make the embdding attend 
to the text about thi:s target specifically and ignore the other parts. this gives the sentiment output for this target specifically
I call this attention module 

> we have got the embedding with all the context of the input sentence and target. now for aspect classification we have simple dense classifier and 
a simple dense regressor for  sentiment score 

> for backprop we have mse loss for score and bce with logits for aspect classification, the aggregated loss is then used for 
updating the weights of the model

> Evaluation results 
