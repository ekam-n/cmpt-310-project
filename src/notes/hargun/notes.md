
# Problems

I thought I successfully trained the model but after viewing its action I realised the model NEVER TURNED.
I spent almost 2 days reading papers and modifying the code until I came to the actual source of the problem
The feature extractor was emmiting nothing but 0's so the model was trying to train off a blalnk screen.
In the effort to solve this solution I tried a lot of other things including

- Switching between a Linear and sequential models.
- I learned from this paper https://arxiv.org/pdf/1511.06581 about how changing when you use the split
    stream Q network matters, and that using a shared stream that "splits" and comes back together 
    stops one of streams from dominating the other

- I tried a few different activation functions from RelU, sigmoid and EMU. I settled on EMU for the shared stream
    and the split streams use ReLu. Theres no reason for this right now other than it seemed to work 

- I tried Different Network architecture sizes from [64,64] - [512,512] I am currently training the working model I got
    on different sized to see how much of a difference it makes

- Getting the advantages of a dueling network doesnt seem to be happening, at around 700,000 timesteps for now which is
    almost the same as DQN without the additional dueling architecture, and this takes longer because of the extra 
    layers and extra necessasary calculation
