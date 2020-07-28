package com.example.generator;

import com.google.common.collect.Lists;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TextGeneratorMain {

    public static final int SEQ_LENGTH = 50;
    public static final int BATCH_SIZE = 8;

    public static void main(String[] args) throws IOException {
        String fileName = "krasnoe-i-chernoe.txt";
        String modelFileName = "model.bin";

        File file = new File(fileName);
        File modelFile = new File(modelFileName);

        AtomicInteger index = new AtomicInteger(0);
        Map<Integer, Integer> uniqueCharsIndicesMap = Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));

        ComputationGraph graph = buildModel(uniqueCharsIndicesMap.keySet().size());
        graph.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage ganStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(ganStatsStorage);
        graph.setListeners(new StatsListener( ganStatsStorage, 20));

        List<Integer> allChars = Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .boxed()
                .collect(Collectors.toList());

        AtomicInteger iteration = new AtomicInteger(0);
        int batchCharsCount = (SEQ_LENGTH + 1) * BATCH_SIZE;
        Lists.partition(allChars, batchCharsCount)
                .stream()
                .filter(batchChars -> batchChars.size() == batchCharsCount)
                .map(batchChars -> batchDataSet(batchChars, uniqueCharsIndicesMap))
                .peek(b -> {
                    if(iteration.incrementAndGet()%1000 == 0 ){
                        try {
                            ModelSerializer.writeModel(graph, modelFile, true);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                })
                .forEach(graph::fit);
    }

    private static DataSet batchDataSet(List<Integer> batchChars, Map<Integer, Integer> uniqueCharsIndicesMap) {
        List<DataSet> batchDataSet = Lists.partition(batchChars, SEQ_LENGTH + 1)
                .stream()
                .map(seqChars -> {
                    List<Integer> firstSubSeq = seqChars.subList(0, SEQ_LENGTH)
                            .stream()
                            .map(uniqueCharsIndicesMap::get)
                            .collect(Collectors.toList());

                    List<Integer> secondSubSeq = seqChars.subList(1, SEQ_LENGTH + 1)
                            .stream()
                            .map(uniqueCharsIndicesMap::get)
                            .collect(Collectors.toList());

                    INDArray indArray = toINDArray(firstSubSeq);
                    INDArray matrix = toMatrix(secondSubSeq, uniqueCharsIndicesMap.size(), SEQ_LENGTH);

                    return new DataSet(indArray, matrix);
                })
                .collect(Collectors.toList());

        return DataSet.merge(batchDataSet);
    }

    private static ComputationGraph buildModel(int uniqueCharsCount) {
        int charEmbedding = 128;
        double learningRate = 0.001;

        int tbpttLength = 50;

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
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

    private static INDArray toINDArray(List<Integer> list) {
        double[] doubles = list.stream().mapToDouble(el -> (double) el).toArray();
        double[][] result = new double[][]{doubles};

        return Nd4j.create(result);
    }

    public static INDArray toMatrix(List<Integer> seq, int size, int maxLength ){

        float[][][] matrix = new float[1][size][maxLength];

        for (int j = 0; j < seq.size(); j++) {
            matrix[0][seq.get(j)][j] = 1;
        }

        return Nd4j.create(matrix);
    }

}
