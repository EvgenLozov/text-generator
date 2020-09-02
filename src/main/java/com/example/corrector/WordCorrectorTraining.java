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
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class WordCorrectorTraining {

    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 50;

    public static void main(String[] args) throws IOException {
        File modelFile = new File("modelCorrector.bin");
        File dictionaryFile = new File("dictonary/words_en_1.txt");

        Map<Integer, Integer> uniqueCharsIndices = uniqueCharsIndices(dictionaryFile);
        List<Character> uniqueChars = uniqueCharsIndices.keySet()
                .stream()
                .map(c -> Character.toChars(c)[0])
                .collect(Collectors.toList());


        List<String> allWords = Files.lines(dictionaryFile.toPath()).collect(Collectors.toList());
        Map<String, Integer> wordsToPositions = wordsToPositions(dictionaryFile);

        ComputationGraph graph = modelFile.exists()
                ? WordsCorrectorModel.load(modelFile)
                : WordsCorrectorModel.build(allWords.size());

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

                                        String misspelledWord = maybeMisspell(word, uniqueChars);

                                        List<Integer> misspelledWordChars = misspelledWord.chars()
                                                .mapToObj(uniqueCharsIndices::get)
                                                .collect(Collectors.toList());

                                        return toDataSet(misspelledWordChars, allWords.size(), position, batchPair.getSecond());
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
        INDArray label = Nd4j.zeros(1, dictionarySize);
        label.put(0, dictionaryPos, 1);

        return label;
    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file) throws IOException {
        AtomicInteger index = new AtomicInteger(0);

        return Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));
    }

    private static Pair<INDArray, INDArray> toFeatureWithMask(List<Integer> list, int maxSize) {
        double[] featureDoubles = list.stream().mapToDouble(el -> (double) el).toArray();
        double[][] result = new double[1][maxSize];

        System.arraycopy(featureDoubles, 0, result[0], 0, featureDoubles.length);

        double[][] mask = new double[1][maxSize];
        for (int i = 0; i < list.size(); i++) {
            mask[0][i] = 1;
        }

        return new Pair<>(Nd4j.create(result), Nd4j.create(mask));
    }

    public static String maybeMisspell(String word, List<Character> chars){
        int letterPos = new Random().nextInt(word.length() - 1) + 1; // do not change the first letter
        StringBuilder sb = new StringBuilder(word);

        switch (new Random().nextInt(3)){
            case 0: {
                sb.deleteCharAt(letterPos);
                break;
            }
            case 1:{
                int charToSet = new Random().nextInt(chars.size());
                sb.setCharAt(letterPos, chars.get(charToSet));
                break;
            }
            case 2: {
                // do nothing
            }
        }

        return sb.toString();
    }
}
