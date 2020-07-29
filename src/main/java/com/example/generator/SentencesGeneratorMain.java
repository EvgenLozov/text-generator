package com.example.generator;

import com.google.common.collect.Lists;
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
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SentencesGeneratorMain {

    private static final int BATCH_SIZE = 128;
    private static final int EPOCHS = 50;

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

        List<List<Integer>> examples = Files.lines(file.toPath())
                .map(line -> line.split("\t")[1])
                .map(line -> trim(line, 50))
                .filter(line -> line.chars().noneMatch(i -> i > 256))
                .map(line -> line.chars().mapToObj(uniqueCharsIndices::get).collect(Collectors.toList()))
                .collect(Collectors.toList());

        AtomicInteger iteration = new AtomicInteger(0);
        IntStream.range(0, EPOCHS)
                .mapToObj(i -> examples)
                .peek(Collections::shuffle)
                .flatMap(shuffledDataSet ->  Lists.partition(shuffledDataSet, BATCH_SIZE).stream())
                .map(batchSeq -> batchSeq.stream()
                                            .map(b -> toDataSet(b, uniqueCharsIndices.size() + 2))
                                            .collect(Collectors.toList()))
                .map(DataSet::merge)
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
        int maxLength = 51;

        List<Integer> featuresSeq = new ArrayList<>(Collections.nCopies(maxLength, endIndex));
        featuresSeq.set(0, startIndex);
        for (int i = 0; i < charsSeq.size(); i++) {
            featuresSeq.set(i+1, charsSeq.get(i));
        }

        List<Integer> labelsSeq = new ArrayList<>(Collections.nCopies(maxLength, endIndex));
        for (int i = 0; i < charsSeq.size(); i++) {
            labelsSeq.set(i, charsSeq.get(i));
        }
        labelsSeq.set(charsSeq.size(), endIndex);

        INDArray features = toINDArray(featuresSeq);
        Pair<INDArray, INDArray> labelsWithMasks = toMatrix(labelsSeq, uniqueCharsCount, maxLength);

        return new DataSet(features, labelsWithMasks.getFirst(), labelsWithMasks.getSecond(), labelsWithMasks.getSecond());
    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file) throws IOException {
        AtomicInteger index = new AtomicInteger(2);

        return Files.lines(file.toPath())
                .filter(line -> line.chars().noneMatch(i -> i > 256))
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
