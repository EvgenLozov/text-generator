package com.example.corrector;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class WordsCorrectorModel {
    public static ComputationGraph load(File modelFile) throws IOException {
        return ModelSerializer.restoreComputationGraph(modelFile);
    }

    public static ComputationGraph build(int uniqueCharsCount) {
        int charEmbedding = 32;
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
                .appendVertex("last_time_step", new LastTimeStepVertex("sequence"))
                .appendLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(charEmbedding)
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
}
