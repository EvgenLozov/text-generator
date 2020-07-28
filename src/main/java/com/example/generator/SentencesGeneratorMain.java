package com.example.generator;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class SentencesGeneratorMain {

    private static final int startIndex = 0;
    private static final int endIndex = 1;

    public static void main(String[] args) throws IOException {
        String examplesFileName = "eng_wikipedia_2010_300K-sentences.txt";
        String modelFileName = "model.bin";

        File modelFile = new File(modelFileName);
        File file = new File(examplesFileName);

        Map<Integer, Integer> uniqueCharsIndices = uniqueCharsIndices(file);

        ComputationGraph graph = Models.sentencesGenerator(uniqueCharsIndices.keySet().size() + 2);
        graph.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage ganStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(ganStatsStorage);
        graph.setListeners(new StatsListener( ganStatsStorage, 20));

        AtomicInteger iteration = new AtomicInteger(0);
        Files.lines(file.toPath())
                .map(line -> line.split("\t")[1])
                .map(line -> trim(line, 50))
                .map(line -> line.chars().mapToObj(uniqueCharsIndices::get).collect(Collectors.toList()))
                .map(seq -> toDataSet(seq, uniqueCharsIndices.size() + 2))
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

    private static DataSet toDataSet(List<Integer> charsSeq, int uniqueCharsCount) {
        List<Integer> featuresSeq = new ArrayList<>();
        featuresSeq.add(startIndex);
        featuresSeq.addAll(charsSeq);

        List<Integer> labelsSeq = new ArrayList<>();
        labelsSeq.addAll(charsSeq);
        labelsSeq.add(endIndex);

        INDArray features = toINDArray(featuresSeq);
        Pair<INDArray, INDArray> labelsWithMasks = toMatrix(labelsSeq, uniqueCharsCount, 51);

        return new DataSet(features, labelsWithMasks.getFirst(), labelsWithMasks.getSecond(), labelsWithMasks.getSecond());
    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file) throws IOException {
        AtomicInteger index = new AtomicInteger(2);

        return Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));
    }

    private static String trim(String line, int limit) {
        if (line.length() <= 50) {
            return line;
        }

        BreakIterator bi = BreakIterator.getWordInstance();
        bi.setText(line);

        return line.substring(0, bi.preceding(limit));
    }

    private static INDArray toINDArray(List<Integer> list) {
        double[] doubles = list.stream().mapToDouble(el -> (double) el).toArray();
        double[][] result = new double[][]{doubles};

        return Nd4j.create(result);
    }

    private static Pair<INDArray,INDArray> toMatrix(List<Integer> seq, int size, int maxLength ){

        float[][][] matrix = new float[1][size][maxLength];
        float[][] mask = new float[1][maxLength];

        for (int j = 0; j < seq.size(); j++) {
            matrix[0][seq.get(j)][j] = 1;
            mask[0][j] = 1;
        }

        return new Pair<>(Nd4j.create(matrix), Nd4j.create(mask));
    }
}
