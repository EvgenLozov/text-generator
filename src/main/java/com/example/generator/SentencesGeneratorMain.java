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
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
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

    private static final int BATCH_SIZE = 64;
    private static final int EPOCHS = 50;

    private static final int START_INDEX = 0;
    private static final int END_INDEX = 1;

    public static final int SENTENCES_LENGTH_LIMIT = 50;

    public static void main(String[] args) throws IOException {
        File modelFile = new File("model.bin");
        File file = new File("eng_wikipedia_2010_300K-sentences.txt");

        Map<Integer, Integer> uniqueCharsIndices = uniqueCharsIndices(file, 2);
        int uniqueCharsCount = uniqueCharsIndices.size() + 2;

        ComputationGraph graph = modelFile.exists()
                ? SentenceGeneratorModel.load(modelFile)
                : SentenceGeneratorModel.build(uniqueCharsCount);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage ganStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(ganStatsStorage);
        graph.setListeners(new StatsListener( ganStatsStorage, 20));

        List<List<Integer>> examples = Files.lines(file.toPath())
                .map(line -> line.split("\t")[1])
                .map(line -> trimSentence(line, SENTENCES_LENGTH_LIMIT))
                .filter(line -> line.chars().noneMatch(i -> i > 256))
                .map(line -> line.chars().mapToObj(uniqueCharsIndices::get).collect(Collectors.toList()))
                .collect(Collectors.toList());

        DataSet scoreDataSet = DataSet.merge(examples.stream()
                                                    .limit(16)
                                                    .map(list -> toDataSet(list, uniqueCharsCount))
                                                    .collect(Collectors.toList()));

        AtomicInteger iteration = new AtomicInteger(1);
        IntStream.range(0, EPOCHS)
                .mapToObj(i -> examples)
                .peek(Collections::shuffle)
                .flatMap(shuffledDataSet ->  Lists.partition(shuffledDataSet, BATCH_SIZE).stream())
                .map(batchSeq -> batchSeq.stream()
                                            .map(b -> toDataSet(b, uniqueCharsCount))
                                            .collect(Collectors.toList()))
                .map(DataSet::merge)
                .peek(b -> {
                    if(iteration.get()%200 == 0 ){
                        try(PrintWriter scoresWriter = new PrintWriter(new FileWriter("scores.txt", true))){
                            scoresWriter.println(graph.score(scoreDataSet));
                        } catch (IOException e){
                            e.printStackTrace();
                        }
                    }
                })
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

    private static DataSet toDataSet(List<Integer> charsSeq, int uniqueCharsCount) {
        int maxLength = SENTENCES_LENGTH_LIMIT + 1;

        List<Integer> featuresSeq = new ArrayList<>(Collections.nCopies(maxLength, END_INDEX));
        featuresSeq.set(0, START_INDEX);
        for (int i = 0; i < charsSeq.size(); i++) {
            featuresSeq.set(i+1, charsSeq.get(i));
        }

        List<Integer> labelsSeq = new ArrayList<>(Collections.nCopies(maxLength, END_INDEX));
        for (int i = 0; i < charsSeq.size(); i++) {
            labelsSeq.set(i, charsSeq.get(i));
        }
        labelsSeq.set(charsSeq.size(), END_INDEX);

        INDArray features = toINDArray(featuresSeq);
        Pair<INDArray, INDArray> labelsWithMasks = toMatrix(labelsSeq, uniqueCharsCount, maxLength);

        return new DataSet(features, labelsWithMasks.getFirst(), labelsWithMasks.getSecond(), labelsWithMasks.getSecond());
    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file, int initialIndex) throws IOException {
        AtomicInteger index = new AtomicInteger(initialIndex);

        return Files.lines(file.toPath())
                .filter(line -> line.chars().noneMatch(i -> i > 256))
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));
    }

    private static String trimSentence(String line, int limit) {
        if (line.length() <= limit) {
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
