package com.example.generator;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

class Models {

    static ComputationGraph sentencesGenerator(int uniqueCharsCount) {
        int charEmbedding = 128;
        double learningRate = 0.001;

        int tbpttLength = 50;

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate, 0, 0.99, 1e-8))
                .graphBuilder()
                .addInputs("sequence")
                .addLayer("noisy_input", new EmbeddingSequenceLayer.Builder()
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .nIn(uniqueCharsCount)
                        .nOut(charEmbedding)
                        .build(), "sequence"
                )
                .appendLayer("lstm", new LSTM.Builder()
                        .nIn(charEmbedding)
                        .nOut(charEmbedding)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build()
                )
                .appendLayer("output", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(128)
                        .nOut(uniqueCharsCount)
                        .build()
                )
//                .backpropType(BackpropType.TruncatedBPTT)
//                .tBPTTForwardLength(tbpttLength)
//                .tBPTTBackwardLength(tbpttLength)
                .setOutputs("output")
                .build();

        return new ComputationGraph(config);
    }

    static ComputationGraph load(File modelFile) throws IOException {
        return ModelSerializer.restoreComputationGraph(modelFile);
    }
}
