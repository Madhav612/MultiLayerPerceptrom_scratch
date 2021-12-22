# Multi-Layer Perceptron from scratch

The approach was simple with this. First, I started implementing activation functions and their derivative fucntions. Then I started working on forward propogation and backward propogation and it was quite challenging as well. I had some difficulty solving the overflow and underflow error so I had to clip values inside each activation function. 

For initialization, I used ha initialization as it leads to better results as it somewhat tackles exploding and vanishing gradient problem. Refered this article:https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

The activation functions and their derivatives were implemented after refering to this article: https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
