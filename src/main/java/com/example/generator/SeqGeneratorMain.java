package com.example.generator;

import org.deeplearning4j.nn.graph.ComputationGraph;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class SeqGeneratorMain {
    public static void main(String[] args) throws IOException {
        File modelFile = new File("model.bin");
        File examplesFile = new File("eng_wikipedia_2010_300K-sentences.txt");

        ComputationGraph graph = SentenceGeneratorModel.load(modelFile);

        Map<Integer, Integer> index2char = uniqueCharsIndices(examplesFile, 2)
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

        Function<Integer, List<Integer>> f = new GeneratorProvider().get(graph, index2char.size())
                .andThen(integers -> integers.stream()
                        .filter(i -> i > 1)
                        .map(index2char::get)
                        .collect(Collectors.toList()));

        for (int i = 0; i < 10; i++) {
            f.apply(0).forEach(c -> System.out.print((char) c.intValue()));
            System.out.println();
            graph.rnnClearPreviousState();
        }

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
}
