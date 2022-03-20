# quora_question_matcher

Repository with a project for Matching and Ranking mini-course final project

This is the implementation of Flask-microservice that allows to search similar question from Quora dataset of questions to user query question.

Algorithm is following:

+ You compute embedding of a question as the average of Glove embeddings of questions' words
+ Than you search for nearest Quora question embeddings by cosine distance measure
+ Then you rank similar questions using KNRM architechture neural network

Microservice use REST API to interact with the user

## How to run the service

In order to run the service you should define the following OS environment variables:

+ `EMB_PATH_GLOVE` - path to glove embeddings file
+ `EMB_PATH_KNRM` - path to KNRM weights
+ `VOCAB_PATH` - path to vocab file
+ `MLP_PATH` - path to mlp trained fro KNRM

Command to run:

`FLASK_APP=user_input.py flask run --port <port_number>`
