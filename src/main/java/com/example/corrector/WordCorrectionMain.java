package com.example.corrector;

import com.example.Pair;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class WordCorrectionMain {
    public static void main(String[] args) throws IOException {
        String wordToCorrect = "educatinal";

        File modelFile = new File("modelCorrector.bin");
        File dictionaryFile = new File("words_en.txt");

        Map<Integer, Integer> uniqueCharsIndices = uniqueCharsIndices(dictionaryFile);

        ComputationGraph graph = WordsCorrectorModel.load(modelFile);

        Map<String, INDArray> wordVectors = Files.lines(dictionaryFile.toPath())
                .collect(Collectors.toMap(Function.identity(), word -> toVector(uniqueCharsIndices, graph, word)));

        INDArray vector = toVector(uniqueCharsIndices, graph, wordToCorrect);

        List<Pair<String, Double>> correctWordCandidates = wordVectors.entrySet()
                .stream()
                .map(candidateEntry -> new Pair<>(candidateEntry.getKey(), vector.sub(candidateEntry.getValue()).norm2(0).getDouble(0)))
                .sorted(Comparator.comparingDouble(Pair::getSecond))
                .limit(3)
                .collect(Collectors.toList());

        System.out.println("Word with a mistake: " + wordToCorrect);
        System.out.println("Correct words candidates: ");
        correctWordCandidates.forEach( w -> System.out.println(w.getFirst()));
    }

    private static INDArray toVector(Map<Integer, Integer> uniqueCharsIndices, ComputationGraph graph, String word) {
        List<Integer> wordChars = word.chars()
                .mapToObj(uniqueCharsIndices::get)
                .collect(Collectors.toList());

        double[] featureDoubles = wordChars.stream().mapToDouble(el -> (double) el).toArray();
        double[][] feature2D = new double[][]{featureDoubles};

        INDArray features = Nd4j.create(feature2D);

        return graph.output(Collections.singletonList("last_time_step"), false, new INDArray[]{features}, null)[0];
    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file) throws IOException {
        AtomicInteger index = new AtomicInteger(0);

        return Files.lines(file.toPath())
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));
    }
}
