package com.example.corrector;

import com.example.Pair;
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
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class WordCorrectorMain {

    private static final int BATCH_SIZE = 64;
    private static final int EPOCHS = 50;

    public static void main(String[] args) throws IOException {
        File modelFile = new File("model.bin");
        File dictionaryFile = new File("words_en.txt");

        Map<Integer, Integer> uniqueCharsIndices = uniqueCharsIndices(dictionaryFile);
        int uniqueCharsCount = uniqueCharsIndices.size();

        List<String> allWords = Files.lines(dictionaryFile.toPath()).collect(Collectors.toList());
        Map<String, Integer> wordsToPositions = wordsToPositions(dictionaryFile);

        ComputationGraph graph = modelFile.exists()
                ? WordsCorrectorModel.load(modelFile)
                : WordsCorrectorModel.build(uniqueCharsCount);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage ganStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(ganStatsStorage);
        graph.setListeners(new StatsListener( ganStatsStorage, 20));

        AtomicInteger iteration = new AtomicInteger(1);
        IntStream.range(0, EPOCHS)
                .mapToObj(i -> allWords)
                .peek(Collections::shuffle)
                .flatMap(shuffledDataSet ->  Lists.partition(shuffledDataSet, BATCH_SIZE).stream())
                .map(batch -> {
                    int maxWordLength = batch.stream().mapToInt(String::length).max().orElse(0);
                    return new Pair<>(batch, maxWordLength);
                })
                .map(batchPair -> batchPair.getFirst()
                                    .stream()
                                    .map(word -> {
                                        Integer position = wordsToPositions.get(word);

                                        List<Integer> wordChars = word.chars()
                                                .mapToObj(uniqueCharsIndices::get)
                                                .collect(Collectors.toList());

                                        return toDataSet(wordChars, allWords.size(), position, batchPair.getSecond());
                                    })
                                    .collect(Collectors.toList()))
                .map(DataSet::merge)
                .peek(b -> {
                    if(iteration.incrementAndGet()%1000 == 0 ){
                        try {
                            System.out.println("Saving model after " + iteration + "-th iteration");
                            ModelSerializer.writeModel(graph, modelFile, true);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                })
                .forEach(graph::fit);

    }

    private static Map<String, Integer> wordsToPositions(File dictionaryFile) throws IOException {
        AtomicInteger index = new AtomicInteger(0);

        return Files.lines(dictionaryFile.toPath())
                .collect(Collectors.toMap(Function.identity(), line -> index.getAndIncrement()));
    }

    private static DataSet toDataSet(List<Integer> seq, int dictonarySize, int dictonaryPos, int wordMaxLength) {
        Pair<INDArray, INDArray> featureWithMask = toFeatureWithMask(seq, wordMaxLength);
        INDArray label = toLabel(dictonarySize, dictonaryPos);

        return new DataSet(featureWithMask.getFirst(), label, featureWithMask.getSecond(), null);
    }

    private static INDArray toLabel(int dictionarySize, int dictionaryPos){
        INDArray label = Nd4j.create(1, dictionarySize);
        label.put(0, dictionaryPos, 1);

        return label;
    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file) throws IOException {
        AtomicInteger index = new AtomicInteger(0);

        return Files.lines(file.toPath())
                .filter(line -> line.chars().noneMatch(i -> i > 256))
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));
    }

    private static Pair<INDArray, INDArray> toFeatureWithMask(List<Integer> list, int maxSize) {
        double[] doubles = list.stream().mapToDouble(el -> (double) el).toArray();
        double[][] result = new double[][]{doubles};

        double[][] mask = new double[1][maxSize];
        for (int i = 0; i < list.size(); i++) {
            mask[0][i] = 1;
        }

        return new Pair<>(Nd4j.create(result), Nd4j.create(mask));
    }


}
