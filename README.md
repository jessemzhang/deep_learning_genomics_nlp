# Learning the Language of the Genome Using RNNs

Epigenetics is the study of how the genome is regulated by external mechanisms.  Biological experiments have shown that subsequences of the human genome are regulated by specific proteins. The purpose of this project is to explore how an RNN architecture can be used to learn sequential patterns in genomic sequences. A robust method for modeling the genome can offer insights on genetic patterns related to health and disease. 

This project was for Stanford's [Deep Learning in NLP](http://cs224d.stanford.edu/) course from Spring 2016.

Please see our [poster](https://github.com/jessemzhang/deep_learning_genomics_nlp/blob/master/project_deliverables/Poster.pdf) and [final report](https://github.com/jessemzhang/deep_learning_genomics_nlp/blob/master/project_deliverables/finalsubmission.pdf) for details. We designed a bidirectional 2-layer RNN for multitask learning using GRUs. You can find code for the TensorFlow model we used [here](https://github.com/jessemzhang/deep_learning_genomics_nlp/blob/master/jz-rnn-tensorflow/enhancer_predictor_lstm_bidirectional_multitask.py).
