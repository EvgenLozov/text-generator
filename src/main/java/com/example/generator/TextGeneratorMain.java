package com.example.generator;

import com.google.common.collect.Lists;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TextGeneratorMain {

    public static final int SEQ_LENGTH = 1000;

    public static void main(String[] args) throws IOException {
        String fileName = "krasnoe-i-chernoe.txt";
        File file = new File(fileName);

        AtomicInteger index = new AtomicInteger(0);
        Map<Integer, Integer> uniqueCharsIndicesMap = Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));

        ComputationGraph graph = buildModel(uniqueCharsIndicesMap.keySet().size());
        graph.init();

        List<Integer> allChars = Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .boxed()
                .collect(Collectors.toList());

        List<List<Integer>> partitions = Lists.partition(allChars, 1001);

        for (List<Integer> partition : partitions) {

            if (partition.size() != SEQ_LENGTH + 1){
                continue;
            }

            List<Integer> firstSubSeq = partition.subList(0, SEQ_LENGTH)
                    .stream()
                    .map(uniqueCharsIndicesMap::get)
                    .collect(Collectors.toList());

            List<Integer> secondSubSeq = partition.subList(1, SEQ_LENGTH + 1)
                    .stream()
                    .map(uniqueCharsIndicesMap::get)
                    .collect(Collectors.toList());

            INDArray indArray = toINDArray(firstSubSeq);
            INDArray matrix = toMatrix(secondSubSeq, uniqueCharsIndicesMap.size(), SEQ_LENGTH);

            // train logic
        }
    }

    private static ComputationGraph buildModel(int uniqueCharsCount) {
        int charEmbedding = 256;
        double learningRate = 0.001;

        int tbpttLength = 50;

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(uniqueCharsCount)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
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
                        .nIn(charEmbedding)
                        .nOut(uniqueCharsCount)
                        .build()
                )
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(tbpttLength)
                .tBPTTBackwardLength(tbpttLength)
                .setOutputs("output")
                .build();

        return new ComputationGraph(config);
    }

    private static INDArray toINDArray(List<Integer> list) {
        double[] doubles = list.stream().mapToDouble(el -> (double) el).toArray();
        return Nd4j.create(doubles);
    }

    public static INDArray toMatrix(List<Integer> seq, int size, int maxLength ){

        float[][][] matrix = new float[1][size][maxLength];

        for (int j = 0; j < seq.size(); j++) {
            matrix[0][seq.get(j)][j] = 1;
        }

        return Nd4j.create(matrix);
    }

}
